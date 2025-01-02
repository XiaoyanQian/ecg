import yaml
import torch
import numpy as np
from PIL import Image
from scipy.io import loadmat
from easydict import EasyDict
from matplotlib.cm import get_cmap
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from models import ECGClip, MultiModalClassifier


def overlay_on_image(ecg_image_path, grad_cams, output_path, alpha=0.5):
    # 1. 加载原始 ECG 图像
    ecg_image = Image.open(ecg_image_path).convert('RGB')
    ecg_image_np = np.array(ecg_image)

    # 2. 定义每条导联的坐标区域（基于更正后的坐标）
    lead_coords = {
        'I': (680, 900, 270, 1660),
        'II': (930, 1080, 270, 1660),
        'III': (1125, 1316, 270, 1660),

        'aVR': (1317, 1585, 270, 1660),
        'aVL': (1600, 1824, 270, 1660),
        'aVF': (1840, 2082, 270, 1660),

        'V1': (680, 900, 1695, 3115),
        'V2': (930, 1080, 1695, 3115),
        'V3': (1125, 1316, 1695, 3115),
        'V4': (1317, 1585, 1695, 3115),
        'V5': (1600, 1824, 1695, 3115),
        'V6': (1840, 2082, 1695, 3115),
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

    print(f'ECG image resolution: {image_height} x {image_width}')
    print(f'Scaled lead coordinates: {lead_coords_scaled}')

    # 4. 结果图像
    blended_image = ecg_image_np.copy()

    # 5. 遍历每条导联的 Grad-CAM
    lead_names = list(lead_coords.keys())  # 确保顺序与 grad_cams 一致
    for lead_idx, grad_cam in enumerate(grad_cams):
        lead_name = lead_names[lead_idx]

        print(f'Processing Lead: {lead_name}')
        print(f'Grad-CAM shape for {lead_name}: {grad_cam.shape}')

        y_start, y_end, x_start, x_end = lead_coords_scaled[lead_name]
        print(
            f'Lead {lead_name}: y=({y_start}, {y_end}), x=({x_start}, {x_end})')
        print(f'Grad-CAM shape: {grad_cam.shape}')

        # Grad-CAM 归一化到 [0, 1]
        grad_cam_normalized = (grad_cam - grad_cam.min()) / \
            (grad_cam.max() - grad_cam.min())
        grad_cam_resized = Image.fromarray(
            (grad_cam_normalized * 255).astype(np.uint8))

        # grad_cam_resized.save(f'grad_cam_{lead_name}.png')
        # print(f'Saved Grad-CAM for {lead_name}')

        grad_cam_resized = grad_cam_resized.resize(
            (x_end - x_start, y_end - y_start), resample=Image.BICUBIC)
        grad_cam_resized_np = np.array(grad_cam_resized)

        # 转换为彩色热图
        cmap = get_cmap('jet')
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


def reshape_ecg(tensor):
    # C,D,W,H
    # return tensor.unsqueeze(2)
    return tensor.reshape(tensor.size(0), tensor.size(1), 10, 10)


def load_ecg(ecg_path):
    ecg = loadmat(ecg_path)['ecg_signals']
    ecg = ecg.astype(np.float32)
    ecg = ecg[:, :5000]
    ecg = (ecg - np.min(ecg))/(np.max(ecg) - np.min(ecg) + 1e-8)
    ecg = torch.from_numpy(ecg).float()
    ecg = ecg.unsqueeze(0)
    return ecg


def load_model(config_path, ckpt_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = EasyDict(yaml.safe_load(f))
    model = ECGClip(config.network)

    # ecg_weights = torch.load(ecg_encoder_path, map_location=device)
    # text_weights = torch.load(text_encoder_path, map_location=device)
    # model.ecg_encoder.load_state_dict(ecg_weights, strict=False)
    # model.lm_model.load_state_dict(text_weights, strict=False)
    weights = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(weights)    
    return model


if __name__ == '__main__':
    device = torch.device('cuda')
    args = EasyDict({
        'target_label': 0,
        'labels': ['LV Impaired'],
        'config_path': 'data/config.yaml',
        'ecg_path': 'data/ecg/mat/1.mat',
        'ecg_image_path': 'data/ecg/image/1.jpg',
        'output_path': 'output/ecg/image/1.png',
        'ckpt': 'data/ckpt/vit_best_ckpt.pth',
        'text_encoder_path': 'data/ckpt/vit_tiny_best_ckpt.pth',
        'ecg_encoder_path': 'data/ckpt/vit_tiny_best_encoder.pth',
    })

    # model: encoder
    model = load_model(args.config_path, args.ckpt)
    model = model.to(device).eval()

    # classifier: labels
    classifier = MultiModalClassifier(model, args.labels, True)
    classifier = classifier.to(device).eval()

    last_block = getattr(classifier.model.ecg_encoder, 'to_patch_embedding')
    target_layers = [last_block]

    ecg_signal = load_ecg(args.ecg_path).to(device)
    ecg_signal = ecg_signal.unsqueeze(2)
    signal_lead_size = ecg_signal.shape[1]

    targets = [ClassifierOutputTarget(args.target_label)]
    grad_cam = GradCAM(model=classifier, target_layers=target_layers, reshape_transform=reshape_ecg)
    input = torch.autograd.variable(ecg_signal.clone(), requires_grad=True)
    grad_cam(input_tensor=input, targets=targets)
    grads = grad_cam.get_in_cam_weights()[0].unsqueeze(1)
    grads = np.maximum(grads.numpy(), 0)
    overlay_on_image(args.ecg_image_path, grads, args.output_path, alpha=0.5)
