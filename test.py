import numpy as np
import torch
import yaml
from torch import nn
from easydict import EasyDict
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from models import ECGClip
from scipy.io import loadmat


class MultiModalClassifier(nn.Module):
    def __init__(self, base_model, labels):
        super().__init__()
        self.model = base_model
        self.labels = labels
        self.use_cam_mode = False  # 添加标志，默认关闭 CAM 模式

    def forward(self, ecg, text=None):
        if ecg.dim() == 4:
            ecg = ecg.squeeze(2)

        if self.use_cam_mode:
            text = [self.labels[0]]

        tokenizer_output = self.model._tokenize(text)
        input_ids = tokenizer_output['input_ids'].to(ecg.device)
        attention_mask = tokenizer_output['attention_mask'].to(ecg.device)
        outputs = self.model(ecg=ecg, input_ids=input_ids, attention_mask=attention_mask)

        proj_ecg_emb = outputs['proj_ecg_emb'][0]
        proj_text_emb = outputs['proj_text_emb'][0]

        logits_per_ecg = proj_ecg_emb@proj_text_emb.T
        if self.use_cam_mode:
            return logits_per_ecg
        else:
            return logits_per_ecg.softmax(dim=-1)

    def forward_for_cam(self, ecg):
        if ecg.dim() == 4:  # 如果输入形状是 [batch_size, channels, 1, length]
            print(f"Input shape before squeeze: {ecg.shape}")
            ecg = ecg.squeeze(2)
            print(f"Input shape after squeeze: {ecg.shape}")

        default_text = [self.labels[0]]  # 假设 labels[0] 对应目标标签
        print(f"Default text input: {default_text}")
        tokenizer_output = self.model._tokenize(default_text)
        input_ids = tokenizer_output['input_ids'].to(ecg.device)
        attention_mask = tokenizer_output['attention_mask'].to(ecg.device)
        outputs = self.model(ecg=ecg, input_ids=input_ids, attention_mask=attention_mask)

        proj_ecg_emb = outputs['ecg_emb']
        if isinstance(proj_ecg_emb, list):
            proj_ecg_emb = proj_ecg_emb[0]

        if proj_ecg_emb.dim() == 2:  # 如果输出是 [B, L]
            proj_ecg_emb = proj_ecg_emb.unsqueeze(1)
        print(f"Output shape: {proj_ecg_emb.shape}")
        return proj_ecg_emb


def preprocess_ecg(ecg_path):
    ecg = loadmat(ecg_path)['ecg_signals']
    ecg = ecg.astype(np.float32)

    ecg = ecg[:, :5000]
    ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)
    ecg = torch.from_numpy(ecg).float()
    ecg = ecg.unsqueeze(0)
    print(f"ECG signal shape: {ecg.shape}")
    return ecg


def reshape_transform_ecg(tensor):
    # 10*10 or 1*100
    result = tensor.reshape(tensor.size(0), 10, 10, tensor.size(2)) # B,Patches,D > B,H,W,D
    return result.transpose(2, 3).transpose(1, 2) # N,D,H,W


def load_yaml_config(config_path):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config file: {e}")


# def load_pretrained_model(config_path, ecg_encoder_path, text_encoder_path, device):
#     config = load_yaml_config(config_path)
#     model = ECGClip(config['network'])
#     print(config['network'])
#     ecg_weights = torch.load(ecg_encoder_path, map_location=device)
#     model.ecg_encoder.load_state_dict(ecg_weights, strict=False)
#     text_weights = torch.load(text_encoder_path, map_location=device)
#     model.lm_model.load_state_dict(text_weights, strict=False)
#     return model.to(device).eval()

def load_pretrained_model(config_path, ecg_encoder_path, device):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = EasyDict(yaml.safe_load(f))
    model = ECGClip(config.network)

    weights = torch.load(ecg_encoder_path, map_location=device)
    model.load_state_dict(weights)
    return model


def overlay_grad_cam_on_image(ecg_image_path, grad_cams, output_path, alpha=0.5):
    """
    将 Grad-CAM 热图叠加到 ECG 图像上，每条导联对应特定的图像区域。
    """
    import numpy as np
    from PIL import Image
    from matplotlib.cm import get_cmap

    # 1. 加载原始 ECG 图像
    ecg_image = Image.open(ecg_image_path).convert("RGB")
    ecg_image_np = np.array(ecg_image)

    # 2. 定义每条导联的坐标区域（基于更正后的坐标）
    lead_coords = {
        "I": (680, 900, 270, 1660),
        "II": (930, 1080, 270, 1660),
        "III": (1125, 1316, 270, 1660),
        "aVR": (1317, 1585, 270, 1660),
        "aVL": (1600, 1824, 270, 1660),
        "aVF": (1840, 2082, 270, 1660),
        "V1": (680, 900, 1695, 3115),
        "V2": (930, 1080, 1695, 3115),
        "V3": (1125, 1316, 1695, 3115),
        "V4": (1317, 1585, 1695, 3115),
        "V5": (1600, 1824, 1695, 3115),
        "V6": (1840, 2082, 1695, 3115),
    }

    # 3. 确保坐标适配图片分辨率
    reference_height = 2480  # 假设参考高度
    reference_width = 3508   # 假设参考宽度
    image_height, image_width, _ = ecg_image_np.shape

    # 根据实际图片大小调整坐标比例
    height_ratio = image_height / reference_height
    width_ratio = image_width / reference_width

    lead_coords_scaled = {
        lead: (
            int(y_start * height_ratio),
            int(y_end * height_ratio),
            int(x_start * width_ratio),
            int(x_end * width_ratio)
        )
        for lead, (y_start, y_end, x_start, x_end) in lead_coords.items()
    }

    print(f"ECG image resolution: {image_height} x {image_width}")
    print(f"Scaled lead coordinates: {lead_coords_scaled}")

    # 4. 结果图像
    blended_image = ecg_image_np.copy()

    # 5. 遍历每条导联的 Grad-CAM
    lead_names = list(lead_coords.keys())  # 确保顺序与 grad_cams 一致
    for lead_idx, grad_cam in enumerate(grad_cams):
        lead_name = lead_names[lead_idx]

        print(f"Processing Lead: {lead_name}")
        print(f"Grad-CAM shape for {lead_name}: {grad_cam.shape}")

        y_start, y_end, x_start, x_end = lead_coords_scaled[lead_name]
        print(f"Lead {lead_name}: y=({y_start}, {y_end}), x=({x_start}, {x_end})")
        print(f"Grad-CAM shape: {grad_cam.shape}")

        # Grad-CAM 归一化到 [0, 1]
        grad_cam_normalized = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
        grad_cam_resized = Image.fromarray((grad_cam_normalized * 255).astype(np.uint8))

        grad_cam_resized.save(f"grad_cam_{lead_name}.png")
        print(f"Saved Grad-CAM for {lead_name}")

        grad_cam_resized = grad_cam_resized.resize((x_end - x_start, y_end - y_start), resample=Image.BICUBIC)
        grad_cam_resized_np = np.array(grad_cam_resized)

        # 转换为彩色热图
        cmap = get_cmap("jet")
        grad_cam_colored = cmap(grad_cam_resized_np / 255.0)[:, :, :3]  # 去掉 alpha 通道
        grad_cam_colored = (grad_cam_colored * 255).astype(np.uint8)

        # 叠加到对应导联区域
        blended_image[y_start:y_end, x_start:x_end, :] = (
            (1 - alpha) * blended_image[y_start:y_end, x_start:x_end, :]
            + alpha * grad_cam_colored
        )

    # 6. 保存并显示结果
    blended_image_pil = Image.fromarray(blended_image.astype(np.uint8))
    blended_image_pil.save(output_path)
    # blended_image_pil.show()
    print(f"Grad-CAM overlay image saved to {output_path}")


def run_ecg_grad_cam(args):
    # load model
    model = load_pretrained_model(args.config_path, args.ecg_encoder_path, args.device)

    classifier = MultiModalClassifier(model, args.labels).to(args.device).eval()
    last_block = getattr(model.ecg_encoder, f'block{model.ecg_encoder.depth - 1}')
    target_layers = [last_block.attn.norm]
    ecg_signal = preprocess_ecg(args.ecg_path).to(args.device)

    ecg_signal = ecg_signal.unsqueeze(2)
    classifier.use_cam_mode = True

    print(f"ECG signal shape before Grad-CAM: {ecg_signal.shape}")
    grad_cams = []  # 存储每个导联的 Grad-CAM 热图
    for lead_idx in range(12):  # 针对每个导联单独计算
        print(f"Processing Lead {lead_idx + 1}")
        # 提取单导联信号
        single_lead_signal = ecg_signal[:, lead_idx:lead_idx+1, :]  # [1, 1, 5000]

        # 创建一个形状为 [1, 12, 5000] 的输入信号，其中其他导联填充为 0
        input_signal = torch.zeros_like(ecg_signal)  # [1, 12, 5000]
        input_signal[:, lead_idx, :] = single_lead_signal[:, 0, :]  # 替换目标导联的信号

        print(f"Input signal shape for Lead {lead_idx + 1}: {input_signal.shape}")

        # Initialize Grad-CAM for this lead
        cam = GradCAM(model=classifier, target_layers=target_layers, reshape_transform=reshape_transform_ecg)
        targets = [ClassifierOutputTarget(args.f)]

        # Compute Grad-CAM for this lead
        grayscale_cam = cam(input_tensor=input_signal, targets=targets)[0]  # 输出形状为 [5000]
        grad_cams.append(grayscale_cam)

    print(f"Generated Grad-CAM for {len(grad_cams)} leads.")
    # 将每个 Grad-CAM 热图叠加到 ECG 图像
    overlay_grad_cam_on_image(args.ecg_image_path, grad_cams, args.output_path, alpha=0.5)


class Args:
    ecg_path = "/home/xyqian/codebase/ecg/data/ecg/mat/1.mat"
    ecg_image_path = "/home/xyqian/codebase/ecg/data/ecg/image/1.jpg"
    labels = ["LV Impaired"]
    config_path = "/home/xyqian/codebase/ecg/data/config.yaml"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    target_label = 0
    output_path = "ecg_grad_cam_overlay.png"
    text_encoder_path = "/home/xyqian/codebase/ecg/data/ckpt/vit_tiny_best_ckpt.pth"
    ecg_encoder_path = "/home/xyqian/codebase/ecg/data/ckpt/vit_tiny_best_ckpt.pth"


if __name__ == '__main__':
    args = Args()
    run_ecg_grad_cam(args)
