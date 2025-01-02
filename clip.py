import os
import cv2
import torch
import numpy as np

from torch import nn
from transformers import CLIPProcessor, CLIPModel
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class Args:
    image_path = "../examples/both.png"  # 替换为你的图像路径
    device = "cuda" if torch.cuda.is_available() else "cpu"
    method = "gradcam"
    output_path = "../output/CLIP_output.jpg"


class ImageClassifier(nn.Module):
    def __init__(self, labels):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.labels = labels

    def forward(self, x):
        text_inputs = self.processor(text=self.labels, return_tensors="pt", padding=True)
        outputs = self.clip(pixel_values=x, input_ids=text_inputs['input_ids'].to(self.clip.device),
                            attention_mask=text_inputs['attention_mask'].to(self.clip.device))
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        for label, prob in zip(self.labels, probs[0]):
            print(f"{label}: {prob:.4f}")
        return probs


def reshape_transform(tensor, height=16, width=16):
    """
    Reshape the Vision Transformer output to match the necessary shape for Grad-CAM.
    """
    tensor = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return tensor.permute(0, 3, 1, 2)  # Rearrange to (batch, channels, height, width)


def run_grad_cam(args):
    # Default labels
    labels = ["a cat", "a dog", "a car", "a person", "a shoe"]

    # Load model and set to evaluation mode
    model = ImageClassifier(labels).to(args.device).eval()

    # Select target layers (ViT's LayerNorm)
    target_layers = [model.clip.vision_model.encoder.layers[-1].layer_norm1]

    # Read and preprocess the image
    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).to(args.device)

    # Initialize Grad-CAM with reshape_transform
    cam_method = GradCAM if args.method == "gradcam" else GradCAMPlusPlus
    cam = cam_method(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    # Use the highest scoring category as default target
    targets = None
    targets = [ClassifierOutputTarget(1)]

    # Generate CAM
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]  # Take the first (and only) image in the batch

    # Overlay CAM on the image
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Save and display the result
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)  # 确保输出目录存在
    cv2.imwrite(args.output_path, cam_image)
    print(f"Grad-CAM result saved to {args.output_path}")


if __name__ == '__main__':
    args = Args()
    run_grad_cam(args)
