import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import cv2
from pathlib import Path
from tqdm import tqdm  # 需要安装: pip install tqdm


class GehlerShiDataset(Dataset):
    def __init__(self, root_dir, mode='train', fold=0, img_size=224, preload=True):
        """
        Args:
            preload (bool): 是否将所有图片加载到内存 (RAM) 以加速训练
        """
        self.root_dir = Path(root_dir)
        self.img_dir = self.root_dir / 'images'
        self.gt_path = self.root_dir / 'real_illum_568.mat'
        self.mode = mode
        self.img_size = img_size
        self.preload = preload

        # --- 1. 加载 GT ---
        if not self.img_dir.exists():
            raise FileNotFoundError(f"找不到目录: {self.img_dir}")

        mat_data = scipy.io.loadmat(str(self.gt_path))
        if 'real_rgb' in mat_data:
            self.gt_illum = mat_data['real_rgb']
        else:
            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            self.gt_illum = mat_data[keys[0]]

        # --- 2. 获取文件列表 ---
        self.image_files = sorted(glob.glob(str(self.img_dir / '*.*')))
        num_data = len(self.image_files)

        # --- 3. 划分 Dataset ---
        fold_size = num_data // 3
        indices = np.arange(num_data)
        if mode == 'train':
            self.indices = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
        else:
            self.indices = indices[fold * fold_size: (fold + 1) * fold_size]

        # --- 4. 高精度预加载 (High-Precision Preloading) ---
        self.ram_cache = {}  # 使用字典存储，key为原始索引

        if self.preload:
            print(f"[{mode.upper()}] 正在预加载 {len(self.indices)} 张高精度 Raw 图片到内存...")
            # 只加载当前 fold 需要的图片，节省内存
            for idx in tqdm(self.indices):
                self.ram_cache[idx] = self._load_image_file(self.image_files[idx])
            print("预加载完成，精度未损失。")
    # ... 在 __init__ 方法结束后 ...

    def __len__(self):
        return len(self.indices)

    # ... 在 __getitem__ 方法之前 ...
    def _load_image_file(self, filepath):
        """读取并保持高精度的辅助函数"""
        # 读取 16-bit 原始数据
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 归一化到 0-1 (Float32)
        if img.dtype == 'uint16':
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32) / 255.0

        # 处理黑电平
        img = np.maximum(img, 0)

        # [关键优化] 限制最大分辨率以防止内存爆炸，但保持足够高的清晰度
        # 对于 224x224 的输入，保留长边 1200-2000 像素绰绰有余
        # 这样既极大减少了内存占用 (比原图小4-10倍)，又保留了所有纹理细节
        MAX_DIM = 1200
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            # INTER_AREA 插值最适合缩小，能保留像素光照能量的平均值
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return img

    def normalize_illumination(self, illum):
        return illum / np.linalg.norm(illum)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]

        # 1. 获取图片数据 (内存 or 硬盘)
        if self.preload:
            img = self.ram_cache[real_idx]
        else:
            img = self._load_image_file(self.image_files[real_idx])

        # 2. 此时 img 已经是 float32 [0, 1] 且分辨率较高 (e.g. 1200x900)
        h, w, c = img.shape

        # 3. 数据增强 (Random Crop)
        if self.mode == 'train':
            crop_h, crop_w = self.img_size, self.img_size
            # 只有当图片真的比 crop 大时才裁剪
            if h > crop_h and w > crop_w:
                x = np.random.randint(0, w - crop_w)
                y = np.random.randint(0, h - crop_h)
                img_patch = img[y:y + crop_h, x:x + crop_w]
            else:
                # 如果图片太小（极少见），直接 Resize
                img_patch = cv2.resize(img, (self.img_size, self.img_size))

            # 翻转
            if np.random.random() > 0.5:
                img_patch = np.flip(img_patch, axis=1).copy()
            if np.random.random() > 0.5:
                img_patch = np.flip(img_patch, axis=0).copy()
        else:
            # 验证集：直接 Resize (或者中心裁剪，这里为了稳定用 Resize)
            img_patch = cv2.resize(img, (self.img_size, self.img_size))

        # 4. GT 处理
        gt = self.gt_illum[real_idx]
        gt = self.normalize_illumination(gt)

        # 5. To Tensor
        img_tensor = torch.from_numpy(img_patch.transpose(2, 0, 1)).float()
        gt_tensor = torch.from_numpy(gt).float()

        return {
            'image': img_tensor,
            'illumination': gt_tensor,
            'filename': os.path.basename(self.image_files[real_idx])
        }


if __name__ == "__main__":
    # 测试预加载速度和内存占用
    import time

    start = time.time()
    ds = GehlerShiDataset(root_dir='./data', mode='train', preload=True)
    print(f"加载耗时: {time.time() - start:.2f}s")
    print(f"第一张图尺寸: {ds.ram_cache[ds.indices[0]].shape}")