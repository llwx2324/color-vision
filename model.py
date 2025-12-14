import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class HybridColorNet(nn.Module):
    def __init__(self, pretrained=True, dropout=0.5):
        """
        CNN + Transformer 混合光照估计网络
        """
        super(HybridColorNet, self).__init__()

        # --- 1. CNN Backbone (SqueezeNet) ---
        # 这是一个非常轻量级的网络，适合小数据集
        squeezenet = models.squeezenet1_1(pretrained=pretrained)

        # 提取特征层 (移除最后的分类器)
        # SqueezeNet 的 features 部分输出为 [N, 512, 13, 13] (当输入为 224x224 时)
        self.features = squeezenet.features

        # 冻结前几层参数? (可选)
        # 只要数据量不是特别少，建议微调所有参数，因为颜色任务和ImageNet分类任务差别很大

        # --- 2. Transformer Neck (创新点) ---
        # 将 CNN 的特征图视为序列: 序列长度 = H*W = 13*13 = 169, 特征维数 = 512
        self.embed_dim = 512
        self.num_heads = 4  # 多头注意力

        # 定义 Transformer 编码层
        # batch_first=True 让输入形状为 [Batch, Seq_Len, Dim]
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=self.num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 位置编码 (Positional Embedding) - 可学习
        # 假设输入最大为 224x224，特征图最大为 13x13=169
        self.pos_embedding = nn.Parameter(torch.randn(1, 169, self.embed_dim))

        # --- 3. Attention Pooling & Regression ---
        # 不直接用 Mean Pooling，而是学习一个权重来聚合 Transformer 的输出
        self.attention_weights = nn.Sequential(
            nn.Linear(self.embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

        # 最终回归层: 512 -> 3 (R, G, B)
        self.regressor = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        # x shape: [Batch, 3, 224, 224]

        # 1. CNN 特征提取
        feat = self.features(x)  # -> [Batch, 512, 13, 13]

        b, c, h, w = feat.shape

        # 2. Reshape 为序列
        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        feat_seq = feat.view(b, c, -1).permute(0, 2, 1)  # -> [B, 169, 512]

        # 3. 添加位置编码
        # 如果输入尺寸变了，这里需要做插值处理，这里暂时假设固定尺寸
        if feat_seq.shape[1] == self.pos_embedding.shape[1]:
            feat_seq = feat_seq + self.pos_embedding

        # 4. Transformer 全局推理
        trans_feat = self.transformer_encoder(feat_seq)  # -> [B, 169, 512]

        # 5. Attention Pooling (加权平均)
        # 计算每个 Patch 的重要性 (比如高光区域更重要)
        attn_scores = self.attention_weights(trans_feat)  # -> [B, 169, 1]

        # 加权求和: sum(feat * weight)
        global_feat = torch.sum(trans_feat * attn_scores, dim=1)  # -> [B, 512]

        # 6. 回归光照值
        pred_illum = self.regressor(global_feat)  # -> [B, 3]

        # 归一化输出 (保证是一个方向向量)
        # 在 Inference 时很重要，训练时有时不归一化反而收敛快，这里我们加上
        pred_illum = F.normalize(pred_illum, p=2, dim=1)

        return pred_illum


# --- 测试代码 ---
if __name__ == "__main__":
    # 模拟一个 Batch 的输入
    dummy_input = torch.randn(2, 3, 224, 224)

    model = HybridColorNet(pretrained=False)  # 测试时不下载权重
    output = model(dummy_input)

    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}")  # 应该是 [2, 3]
    print(f"输出示例: {output[0].detach().numpy()}")

    # 打印参数量 (凑 PPT 字数用)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f} M")