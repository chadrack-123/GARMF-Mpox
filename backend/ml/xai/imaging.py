"""
XAI (Explainable AI) module for imaging models using Grad-CAM
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple
import logging
import cv2

logger = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM implementation for PyTorch models"""
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM
        
        Args:
            model: PyTorch model
            target_layer: Layer to compute gradients on
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save activations"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap
        
        Args:
            input_tensor: Input image tensor
            target_class: Target class (None for predicted class)
            
        Returns:
            Heatmap as numpy array
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass
        target = output[0, target_class]
        target.backward()
        
        # Compute Grad-CAM
        gradients = self.gradients[0]  # Shape: (C, H, W)
        activations = self.activations[0]  # Shape: (C, H, W)
        
        # Global average pooling on gradients
        weights = gradients.mean(dim=(1, 2))  # Shape: (C,)
        
        # Weighted combination of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = torch.clamp(cam, min=0)
        
        # Normalize
        cam = cam / (cam.max() + 1e-10)
        
        return cam.cpu().numpy()
    
    def visualize(
        self,
        image_path: Path,
        heatmap: np.ndarray,
        output_dir: Path,
        alpha: float = 0.4
    ) -> Tuple[Path, Path]:
        """
        Visualize Grad-CAM heatmap
        
        Args:
            image_path: Path to original image
            heatmap: Grad-CAM heatmap
            output_dir: Output directory
            alpha: Overlay transparency
            
        Returns:
            Tuple of (heatmap_path, overlay_path)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load original image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
        
        # Convert to colormap
        heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Save heatmap
        heatmap_path = output_dir / f"{image_path.stem}_heatmap.png"
        Image.fromarray(heatmap_colored).save(heatmap_path)
        
        # Create overlay
        overlay = (alpha * heatmap_colored + (1 - alpha) * image_np).astype(np.uint8)
        
        # Save overlay
        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        Image.fromarray(overlay).save(overlay_path)
        
        logger.info(f"Saved Grad-CAM visualizations to {output_dir}")
        return heatmap_path, overlay_path


def generate_gradcam_for_model(
    model: nn.Module,
    image_path: Path,
    output_dir: Path,
    device: torch.device,
    image_size: int = 224,
    model_name: str = "resnet18"
) -> dict:
    """
    Generate Grad-CAM for a single image
    
    Args:
        model: Trained PyTorch model
        image_path: Path to input image
        output_dir: Output directory
        device: Torch device
        image_size: Image size
        model_name: Model architecture name
        
    Returns:
        Dictionary with paths to generated files
    """
    # Get target layer
    if "resnet" in model_name:
        target_layer = model.layer4[-1]
    elif "densenet" in model_name:
        target_layer = model.features.denseblock4
    elif "vit" in model_name:
        # ViT doesn't work well with Grad-CAM
        logger.warning("ViT models are not ideal for Grad-CAM")
        target_layer = model.encoder.layers[-1]
    else:
        raise ValueError(f"Unknown model for Grad-CAM: {model_name}")
    
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate heatmap
    heatmap = grad_cam.generate(input_tensor)
    
    # Visualize
    heatmap_path, overlay_path = grad_cam.visualize(image_path, heatmap, output_dir)
    
    return {
        "image": str(image_path),
        "heatmap": str(heatmap_path),
        "overlay": str(overlay_path)
    }
