import cv2
import torch
import numpy as np
from model import HybridColorNet

# --- 配置 ---
MODEL_PATH = 'best_model.pth'
INPUT_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_GREEN_BIAS = 1.70

# 全局模拟参数
SIM_LIGHT = {"R": 1.0, "G": 1.0, "B": 1.0}


class ColorDetector:
    # ... (ColorDetector 类保持不变，直接复制之前的即可) ...
    def __init__(self):
        self.colors_hsv = {
            "Red": [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [179, 255, 255])],
            "Orange": [([11, 100, 100], [25, 255, 255])],
            "Yellow": [([26, 100, 100], [34, 255, 255])],
            "Green": [([35, 80, 80], [85, 255, 255])],
            "Cyan": [([80, 100, 100], [95, 255, 255])],
            "Blue": [([100, 100, 80], [130, 255, 255])],
            "Purple": [([135, 50, 50], [155, 255, 255])],
            "Pink": [([145, 50, 100], [170, 255, 255])],
            "White": [([0, 0, 180], [180, 30, 255])],
            "Gray": [([0, 0, 50], [180, 50, 179])],
        }
        self.draw_colors = {
            "Red": (0, 0, 255), "Orange": (0, 140, 255), "Yellow": (0, 255, 255),
            "Green": (0, 255, 0), "Cyan": (255, 255, 0), "Blue": (255, 0, 0),
            "Purple": (226, 43, 138), "Pink": (180, 105, 255), "White": (255, 255, 255),
            "Gray": (128, 128, 128)
        }

    def detect(self, image_bgr, min_area=800):
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        output_img = image_bgr.copy()
        detected_results = []
        for color_name, ranges in self.colors_hsv.items():
            mask = np.zeros(hsv.shape[:2], dtype="uint8")
            for (lower, upper) in ranges:
                lower = np.array(lower, dtype="uint8")
                upper = np.array(upper, dtype="uint8")
                mask += cv2.inRange(hsv, lower, upper)
            mask = cv2.erode(mask, None, iterations=1)
            mask = cv2.dilate(mask, None, iterations=2)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area > min_area:
                    x, y, w, h = cv2.boundingRect(c)
                    detected_results.append({"color": color_name, "area": area, "rect": (x, y, w, h)})
        detected_results.sort(key=lambda x: x["area"], reverse=True)
        for item in detected_results[:6]:
            x, y, w, h = item["rect"]
            name = item["color"]
            bgr = self.draw_colors.get(name, (0, 255, 0))
            cv2.rectangle(output_img, (x, y), (x + w, y + h), bgr, 2)
            cv2.putText(output_img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(output_img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 1)
        return output_img


def setup_camera(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    return cap


def apply_wb_correction(img_linear, pred_illum, green_bias):
    """
    动态 Bias 版校正函数
    """
    # 1. 基础增益计算
    g_val = pred_illum[1]
    r_pred = max(pred_illum[0], 0.05)
    b_pred = max(pred_illum[2], 0.05)

    r_gain = g_val / r_pred
    b_gain = g_val / b_pred

    # --- 核心修改：动态调整 Green Bias ---
    # 逻辑：如果模型检测到环境光非常绿 (g_val > 0.6)，说明环境里绿色已经够多了，
    # 这时候就不应该再强行乘以 1.7 的 Bias，否则会矫枉过正。
    # 我们设计一个简单的衰减函数。

    current_bias = green_bias

    # 阈值设定：通常 G 在归一化向量里是 0.57 左右 (1/sqrt(3))。
    # 如果 G > 0.65，说明显著偏绿。
    if g_val > 0.65:
        # 计算超出的程度
        excess = (g_val - 0.65) / 0.35  # 归一化到 0~1
        # 线性插值：环境越绿，Bias 越接近 1.0 (即不再额外补偿)
        # 当 excess=0 时 bias=1.7; 当 excess=1 时 bias=1.0
        current_bias = green_bias * (1 - excess) + 1.0 * excess

        # 打印一下方便调试 (可选)
        # print(f"Auto-reducing bias to {current_bias:.2f} due to strong green light")

    g_gain = current_bias

    # 2. 归一化防止过曝 (保持不变)
    max_gain = max(r_gain, g_gain, b_gain)
    scale = 0.90 / (max_gain + 1e-6)

    r_gain *= scale
    g_gain *= scale
    b_gain *= scale

    img_corrected = img_linear.copy()
    img_corrected[:, :, 0] *= r_gain
    img_corrected[:, :, 1] *= g_gain
    img_corrected[:, :, 2] *= b_gain

    # 3. 自动曝光补偿 (保持不变)
    current_max = np.percentile(img_corrected, 95)
    if current_max > 0.001:
        target = 0.85
        auto = np.clip(target / current_max, 0.5, 3.0)
        img_corrected *= auto

    return np.clip(img_corrected, 0, 1)

def main():
    global SIM_LIGHT
    print("Running Final Fix Demo...")

    model = HybridColorNet(pretrained=False)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except:
        return
    model.to(DEVICE)
    model.eval()

    detector = ColorDetector()
    cap = setup_camera(0)

    smooth_illum = None
    alpha = 0.1

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- 步骤 1: 获取纯净的线性图 ---
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_linear_base = np.power(img_rgb, 2.2)

        # --- 步骤 2: 施加模拟光照 (并且保留颜色比率) ---
        img_sim = img_linear_base.copy()
        img_sim[:, :, 0] *= SIM_LIGHT["R"]
        img_sim[:, :, 1] *= SIM_LIGHT["G"]
        img_sim[:, :, 2] *= SIM_LIGHT["B"]

        # 防止截断导致变白 (Desaturation)
        # 如果某个像素值超过1.0，整体缩小该像素，保持 R:G:B 比例不变
        max_per_pixel = np.max(img_sim, axis=2, keepdims=True)
        # 只缩放那些溢出的像素
        scale_mask = np.maximum(max_per_pixel, 1.0)
        img_sim = img_sim / scale_mask

        # --- 步骤 3: 混合模型预测 (Hybrid Prediction) ---
        # 这是让 Demo 成功的关键 Trick
        # 既然我们知道这是一个极端环境，单一模型可能不准。
        # 我们可以计算一下简单的“灰度世界假设”(Gray World) 均值，作为参考。

        mean_r = np.mean(img_sim[:, :, 0])
        mean_g = np.mean(img_sim[:, :, 1])
        mean_b = np.mean(img_sim[:, :, 2])
        norm = np.sqrt(mean_r ** 2 + mean_g ** 2 + mean_b ** 2) + 1e-6
        gw_pred = np.array([mean_r / norm, mean_g / norm, mean_b / norm])

        # AI 模型推理
        img_small = cv2.resize(img_sim, (INPUT_SIZE, INPUT_SIZE))
        input_tensor = torch.from_numpy(img_small.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            ai_pred = model(input_tensor).cpu().numpy()[0]

        # 【核心修正】
        # 很多时候，在极端色偏下，传统的 Gray World 其实比 AI 更准（虽然在复杂光照下 AI 准）。
        # 为了 Demo 效果，我们将 AI 预测和 Gray World 预测融合。
        # 正常情况下 AI 占主导；极端情况下 Gray World 帮忙修正方向。

        # 简单的 50/50 融合，或者 70% AI + 30% GW
        # 这样能保证 R 大的时候，预测向量里的 R 肯定也是大的
        final_pred = 0.6 * ai_pred + 0.4 * gw_pred

        # 归一化
        final_pred = final_pred / np.linalg.norm(final_pred)

        # 平滑
        if smooth_illum is None:
            smooth_illum = final_pred
        else:
            smooth_illum = (1 - alpha) * smooth_illum + alpha * final_pred

        # --- 步骤 4: 校正 ---
        img_corr = apply_wb_correction(img_sim, smooth_illum, green_bias=BEST_GREEN_BIAS)

        # --- 步骤 5: 显示 ---
        # 左图：模拟图
        sim_srgb = np.power(img_sim, 1 / 2.2)
        sim_bgr = cv2.cvtColor(np.clip(sim_srgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # 右图：结果图
        corr_srgb = np.power(img_corr, 1 / 2.2)
        corr_bgr = cv2.cvtColor(np.clip(corr_srgb * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # 颜色检测
        left_res = detector.detect(sim_bgr)
        right_res = detector.detect(corr_bgr)

        h, w = frame.shape[:2]
        display_h = 480
        scale = display_h / h
        display_w = int(w * scale)
        combined = np.hstack((
            cv2.resize(left_res, (display_w, display_h)),
            cv2.resize(right_res, (display_w, display_h))
        ))

        # 绘制信息
        text_pred = f"Pred: R{smooth_illum[0]:.2f} G{smooth_illum[1]:.2f} B{smooth_illum[2]:.2f}"
        cv2.putText(combined, text_pred, (display_w + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(combined, "Simulated (Input)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(combined, "Corrected (Result)", (display_w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        status = f"Simulated Gain -> R:{SIM_LIGHT['R']:.1f} G:{SIM_LIGHT['G']:.1f} B:{SIM_LIGHT['B']:.1f}"
        cv2.putText(combined, status, (10, display_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow('Final Demo', combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            SIM_LIGHT = {"R": 1.0, "G": 1.0, "B": 1.0}
        elif key == ord('r'):
            SIM_LIGHT["R"] += 0.2
        elif key == ord('t'):
            SIM_LIGHT["R"] = max(0.2, SIM_LIGHT["R"] - 0.2)
        elif key == ord('g'):
            SIM_LIGHT["G"] += 0.2
        elif key == ord('h'):
            SIM_LIGHT["G"] = max(0.2, SIM_LIGHT["G"] - 0.2)
        elif key == ord('b'):
            SIM_LIGHT["B"] += 0.2
        elif key == ord('n'):
            SIM_LIGHT["B"] = max(0.2, SIM_LIGHT["B"] - 0.2)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()