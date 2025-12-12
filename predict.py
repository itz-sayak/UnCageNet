#predict.py

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
from tqdm import tqdm
import datetime
import subprocess
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Configure PyTorch memory allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    found_4090 = False
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        if "4090" in name:
            device = torch.device(f"cuda:{i}")
            torch.cuda.set_device(device)
            print(f"Using RTX 4090 at cuda:{i} ({name})")
            found_4090 = True
            break
    if not found_4090:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print(f"RTX 4090 not found, using cuda:0 ({torch.cuda.get_device_name(0)})")
    # Report memory usage
    mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
    mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    mem_total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
    print(f"GPU memory: allocated {mem_alloc:.2f} GB, reserved {mem_reserved:.2f} GB, total {mem_total:.2f} GB")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Helper function for orientation visualization
def get_ori_map(orient):
    """Convert orientation to color visualization"""
    orient = orient[...,None] 
    cos = np.cos(orient)
    orient_img = np.concatenate([cos * (cos >= 0), np.sin(orient), np.abs(cos) * (cos < 0)], axis=-1)
    orient_img = (orient_img * 255).round().astype('uint8')
    return orient_img

# Morphological dilation function
def dilate_mask(mask, kernel_size=3, iterations=1):
    """Apply morphological dilation to a binary mask using OpenCV."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=iterations)
    return dilated_mask

# Gabor Filter Module
class calOrientationGabor(nn.Module):
    def __init__(self, channel_in=1, channel_out=1, stride=1):
        super(calOrientationGabor, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.numKernels = 72
        self.clamp_confidence_low = 0.0
        self.clamp_confidence_high = 0.2
        self.sigma_x = nn.Parameter(torch.ones(1) * 1.8)
        self.sigma_y = nn.Parameter(torch.ones(1) * 2.4)
        self.Lambda = nn.Parameter(torch.ones(1) * 4.0)
        self.kernel_size = 17
        
    def filter(self, image, label, threshold, variance_data, orient_data, max_resp_data):
        batch_size = 6
        H, W = image.size()[2:4]
        all_responses = []
        all_orients = []
        
        for batch_start in range(0, self.numKernels, batch_size):
            batch_end = min(batch_start + batch_size, self.numKernels)
            batch_resArray = []
            batch_orients = []
            
            for iOrient in range(batch_start, batch_end):
                theta = nn.Parameter(torch.ones(self.channel_out) * (math.pi * iOrient / self.numKernels)).to(device)
                GaborKernel = self.gabor_fn(self.kernel_size, self.channel_in, self.channel_out, theta, 
                                            self.sigma_x, self.sigma_y, self.Lambda)
                response = F.conv2d(image, GaborKernel, padding=self.kernel_size//2)
                batch_resArray.append(response)
                orient_tensor = torch.ones(1, 1, H, W).to(device) * math.pi * iOrient / self.numKernels
                batch_orients.append(orient_tensor)
            
            all_responses.extend(batch_resArray)
            all_orients.extend(batch_orients)
            torch.cuda.empty_cache()
        
        resTensor = torch.cat(all_responses, dim=1)
        orient = torch.cat(all_orients, dim=1)
        resTensor = torch.abs(resTensor)
        max_resp = torch.max(resTensor, dim=1, keepdim=True)[0]
        maxResTensor = torch.argmax(resTensor, dim=1, keepdim=True).float()
        best_orientTensor = maxResTensor * math.pi / self.numKernels

        orient_diff = torch.minimum(torch.abs(best_orientTensor - orient),
                           torch.minimum(torch.abs(best_orientTensor - orient - math.pi),
                                      torch.abs(best_orientTensor - orient + math.pi)))

        resp_diff = resTensor - max_resp
        variance = torch.sum(orient_diff * resp_diff * resp_diff, dim=1, keepdim=True)
        variance = variance ** (1 / 2)
        
        orient_data = torch.where(variance > variance_data, best_orientTensor, orient_data)
        max_resp_data = torch.where(variance > variance_data, max_resp, max_resp_data)
        variance_data = torch.where(variance > variance_data, variance, variance_data)

        max_all_resp = torch.max(max_resp_data)
        max_all_var = torch.max(variance_data)
        
        if max_all_resp > 0:
            max_resp_data = max_resp_data / max_all_resp
        if max_all_var > 0:
            variance_data = variance_data / max_all_var

        confidenceTensor = (variance_data - self.clamp_confidence_low) / (
                        self.clamp_confidence_high - self.clamp_confidence_low)
        confidenceTensor = confidenceTensor.clamp(0, 1)

        del resTensor, orient, max_resp, maxResTensor, best_orientTensor, orient_diff, resp_diff, variance
        torch.cuda.empty_cache()

        return confidenceTensor, variance_data, orient_data
        
    def forward(self, image, label, iter=1, threshold=0.0):
        H, W = image.size()[2:4]
        variance_data = torch.ones(1, 1, H, W).to(device) * 0
        orient_data = torch.ones(1, 1, H, W).to(device) * 0
        max_resp_data = torch.ones(1, 1, H, W).to(device) * 0

        for i in range(iter):
            confidenceTensor, variance_data, orient_data = self.filter(
                image, label, threshold, variance_data, orient_data, max_resp_data
            )
            image = confidenceTensor
        
        confidenceTensor[confidenceTensor < threshold] = 0
        best_orientTensor = orient_data
        orientTwoChannel = torch.cat([torch.sin(best_orientTensor), torch.cos(best_orientTensor)], dim=1)
        return orientTwoChannel, best_orientTensor, confidenceTensor

    def gabor_fn(self, kernel_size, channel_in, channel_out, theta, sigma_x, sigma_y, Lambda, phase=0.):
        sigma_x_expanded = sigma_x.expand(channel_out)
        sigma_y_expanded = sigma_y.expand(channel_out)
        Lambda_expanded = Lambda.expand(channel_out)
        psi = nn.Parameter(torch.ones(channel_out) * phase).to(device)

        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax
        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).to(device).float() - 0.5
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
        x_0 = torch.arange(xmin, xmax + 1).to(device).float() - 0.5
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()  

        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

        gb = torch.exp(
            -.5 * (x_theta ** 2 / sigma_x_expanded.view(-1, 1, 1, 1) ** 2 + 
                   y_theta ** 2 / sigma_y_expanded.view(-1, 1, 1, 1) ** 2)) \
             * torch.cos(2 * math.pi * x_theta / Lambda_expanded.view(-1, 1, 1, 1) + psi.view(-1, 1, 1, 1))

        return gb

# ResNet Encoder
class ResNetEncoder(nn.Module):
    def __init__(self, backbone='resnet34', pretrained=True, input_channels=3):
        super(ResNetEncoder, self).__init__()
        
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
            self.feature_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=pretrained)
            self.feature_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
            self.feature_channels = [64, 256, 512, 1024, 2048]
        elif backbone == 'resnet101':
            self.resnet = models.resnet101(pretrained=pretrained)
            self.feature_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        self.backbone = backbone
        self.input_channels = input_channels
        self.resnet.fc = nn.Identity()
        
        if input_channels != 3:
            original_conv1 = self.resnet.conv1
            new_conv1 = nn.Conv2d(
                input_channels, 
                original_conv1.out_channels,
                kernel_size=original_conv1.kernel_size,
                stride=original_conv1.stride,
                padding=original_conv1.padding,
                bias=original_conv1.bias is not None
            )
            
            with torch.no_grad():
                if input_channels > 3:
                    new_conv1.weight[:, :3, :, :] = original_conv1.weight
                    nn.init.kaiming_normal_(new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')
                    new_conv1.weight[:, 3:, :, :] *= 0.1
                elif input_channels < 3:
                    new_conv1.weight = nn.Parameter(original_conv1.weight[:, :input_channels, :, :])
                
                if original_conv1.bias is not None:
                    new_conv1.bias = nn.Parameter(original_conv1.bias)
            
            self.resnet.conv1 = new_conv1
        
        self.conv1 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.maxpool = self.resnet.maxpool
        self.layer1 = self.resnet.layer1
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4
    
    def forward(self, x):
        features = []
        x = self.conv1(x)
        features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return features

# Double Convolution Block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if use_batchnorm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

# UNet with ResNet Backbone and Gabor Features
class UNetWithResNetBackbone(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, backbone='resnet101', 
                 decoder_channels=[256, 128, 64, 32, 16], use_gabor=True, pretrained=True):
        super(UNetWithResNetBackbone, self).__init__()
        self.use_gabor = use_gabor
        self.backbone = backbone
        
        total_input_channels = in_channels
        if self.use_gabor:
            self.gabor_filter = calOrientationGabor(channel_in=1, channel_out=1)
            total_input_channels += 2
        
        self.encoder = ResNetEncoder(backbone=backbone, pretrained=pretrained, 
                                   input_channels=total_input_channels)
        encoder_channels = self.encoder.feature_channels
        self.decoder_blocks = nn.ModuleList()
        self.center = DoubleConv(encoder_channels[-1], decoder_channels[0])
        
        for i in range(len(decoder_channels)):
            if i == 0:
                in_ch = decoder_channels[0]
                skip_ch = encoder_channels[-2]
                out_ch = decoder_channels[0]
            else:
                in_ch = decoder_channels[i-1]
                if len(encoder_channels) - 2 - i >= 0:
                    skip_ch = encoder_channels[len(encoder_channels) - 2 - i]
                else:
                    skip_ch = 0
                out_ch = decoder_channels[i]
            
            self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
        
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], out_channels, kernel_size=1)

    def forward(self, x, store_intermediate=False):
        original_size = x.shape[2:]
        
        # Initialize storage for intermediate results
        if store_intermediate:
            self.intermediate_results = {}
            self.intermediate_results['original'] = x.detach().clone()
        
        if self.use_gabor:
            gray = torch.mean(x, dim=1, keepdim=True)
            dummy_label = torch.ones_like(gray)
            orientTwoChannel, best_ori, confidence = self.gabor_filter(gray, dummy_label)
            
            confidence_threshold = 0.1
            bconf = best_ori * (confidence > confidence_threshold)
            bconfTwoChannel = torch.cat([torch.sin(bconf), torch.cos(bconf)], dim=1)
            x = torch.cat([x, bconfTwoChannel], dim=1)
            
            # Store intermediate Gabor results
            if store_intermediate:
                self.intermediate_results['grayscale'] = gray.detach().clone()
                self.intermediate_results['orientation'] = best_ori.detach().clone()
                self.intermediate_results['confidence'] = confidence.detach().clone()
                self.intermediate_results['masked_orientation'] = bconf.detach().clone()
            
            self.confidence_map = confidence.detach()
            del gray, dummy_label, orientTwoChannel, best_ori, confidence, bconf, bconfTwoChannel
            torch.cuda.empty_cache()
        
        encoder_features = self.encoder(x)
        center = self.center(encoder_features[-1])
        x = center
        
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_idx = len(encoder_features) - 2 - i
            if skip_idx >= 0:
                skip_feature = encoder_features[skip_idx]
            else:
                skip_feature = None
            x = decoder_block(x, skip_feature)
        
        x = self.segmentation_head(x)
        
        if x.shape[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=True)
        
        # Store final output
        if store_intermediate:
            self.intermediate_results['raw_output'] = x.detach().clone()
        
        return x
    
    def get_enhanced_mask(self, x, confidence_boost=0.4):
        """Get confidence-enhanced mask"""
        with torch.no_grad():
            base_output = self.forward(x)
            base_probs = torch.sigmoid(base_output)
            
            if self.use_gabor and hasattr(self, 'confidence_map'):
                confidence_resized = F.interpolate(
                    self.confidence_map, size=base_output.shape[2:], mode='bilinear', align_corners=True
                )
                enhanced_probs = base_probs + (confidence_resized * confidence_boost)
                enhanced_probs = torch.clamp(enhanced_probs, 0, 1)
                return enhanced_probs
            
            return base_probs

# Dataset class
class ImageDataset(Dataset):
    def __init__(self, images_dir, target_size=(256, 144)):
        self.images_dir = images_dir
        self.target_size = target_size
        self.images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif'))]
        print(f"Found {len(self.images)} images for prediction")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error loading image: {img_path}")
            blank = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            return torch.from_numpy(blank.transpose(2, 0, 1)).float(), img_name
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Store original size for later use
        original_size = (image.shape[1], image.shape[0])  # (width, height)
        image = cv2.resize(image, self.target_size)
        image = image / 255.0
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        
        return image_tensor, img_name, original_size

def create_exact_overlay_from_saved_mask(original_image_path, mask_path, target_size, alpha=0.6):
    """Create red overlay using EXACT saved mask on original image"""
    # Load original image at full resolution
    original = cv2.imread(original_image_path)
    if original is None:
        print(f"Could not load original image: {original_image_path}")
        return None
    
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    original_size = (original.shape[1], original.shape[0])  # (width, height)
    
    # Load saved mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not load mask: {mask_path}")
        return None
    
    # Resize mask to match original image size (not target size!)
    mask_resized = cv2.resize(mask, original_size)
    
    # Create red overlay where mask is white (>127)
    overlay = original.copy()
    overlay[mask_resized > 127] = [255, 0, 0]  # Red color in RGB
    
    # Blend original and overlay
    result = cv2.addWeighted(original, 1-alpha, overlay, alpha, 0)
    
    return result

# Complete Pipeline Visualization Function
def visualize_complete_pipeline(model, input_tensor, filename, save_dir, threshold=0.3, confidence_boost=0.4,
                               apply_dilation=False, dilate_kernel_size=3, dilate_iterations=1):
    """
    Visualize the complete processing pipeline from input to final result
    """
    print(f"Creating complete pipeline visualization for {filename}...")
    
    # Create pipeline visualization directory
    pipeline_dir = os.path.join(save_dir, 'pipeline_visualization')
    os.makedirs(pipeline_dir, exist_ok=True)
    
    with torch.no_grad():
        # Forward pass with intermediate storage
        model.eval()
        output = model.forward(input_tensor, store_intermediate=True)
        
        # Get enhanced mask
        enhanced_mask = model.get_enhanced_mask(input_tensor, confidence_boost)
        
        # Extract intermediate results
        intermediate = model.intermediate_results
        
        # Convert tensors to numpy for visualization
        original_img = intermediate['original'][0].permute(1, 2, 0).cpu().numpy()
        grayscale = intermediate['grayscale'][0, 0].cpu().numpy()
        orientation = intermediate['orientation'][0, 0].cpu().numpy()
        confidence = intermediate['confidence'][0, 0].cpu().numpy()
        masked_orientation = intermediate['masked_orientation'][0, 0].cpu().numpy()
        raw_output = torch.sigmoid(intermediate['raw_output'][0, 0]).cpu().numpy()
        enhanced_output = enhanced_mask[0, 0].cpu().numpy()
        
        # Create orientation color map
        orientation_color = get_ori_map(orientation)
        masked_orientation_color = get_ori_map(masked_orientation)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle(f'Complete Processing Pipeline: {filename}', fontsize=16, fontweight='bold')
        
        # Row 1: Input Processing
        axes[0, 0].imshow(original_img)
        axes[0, 0].set_title('1. Original Input Image', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(grayscale, cmap='gray')
        axes[0, 1].set_title('2. Grayscale Conversion', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(orientation_color)
        axes[0, 2].set_title('3. Gabor Orientation Map', fontweight='bold')
        axes[0, 2].axis('off')
        
        conf_im = axes[0, 3].imshow(confidence, cmap='jet', vmin=0, vmax=1)
        axes[0, 3].set_title('4. Confidence Map', fontweight='bold')
        axes[0, 3].axis('off')
        plt.colorbar(conf_im, ax=axes[0, 3], fraction=0.046, pad=0.04)
        
        # Row 2: Processing Steps
        axes[1, 0].imshow(masked_orientation_color)
        axes[1, 0].set_title('5. Masked Orientation\n(High Confidence Only)', fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(raw_output, cmap='gray', vmin=0, vmax=1)
        axes[1, 1].set_title('6. Base UNet Prediction', fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(enhanced_output, cmap='gray', vmin=0, vmax=1)
        axes[1, 2].set_title('7. Confidence-Enhanced Mask', fontweight='bold')
        axes[1, 2].axis('off')
        
        # Enhancement difference
        enhancement_diff = enhanced_output - raw_output
        diff_im = axes[1, 3].imshow(enhancement_diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
        axes[1, 3].set_title('8. Enhancement Difference', fontweight='bold')
        axes[1, 3].axis('off')
        plt.colorbar(diff_im, ax=axes[1, 3], fraction=0.046, pad=0.04)
        
        # Row 3: Final Results and Overlays
        binary_mask = (enhanced_output > threshold).astype(np.float32)
        binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
        
        # Apply dilation if requested
        if apply_dilation:
            dilated_mask_uint8 = dilate_mask(binary_mask_uint8, dilate_kernel_size, dilate_iterations)
            dilated_mask = dilated_mask_uint8.astype(np.float32) / 255.0
        else:
            dilated_mask = binary_mask
        
        axes[2, 0].imshow(dilated_mask, cmap='gray')
        title = f'9. Binary Mask\n(Threshold: {threshold}'
        if apply_dilation:
            title += f', Dilated: {dilate_kernel_size}x{dilate_iterations})'
        else:
            title += ')'
        axes[2, 0].set_title(title, fontweight='bold')
        axes[2, 0].axis('off')
        
        # Create overlay using dilated mask
        overlay = original_img.copy()
        overlay[dilated_mask > 0.5] = [1, 0, 0]  # Red color
        blended_overlay = cv2.addWeighted((original_img * 255).astype(np.uint8), 
                                         0.6, (overlay * 255).astype(np.uint8), 0.4, 0)
        blended_overlay = blended_overlay.astype(np.float32) / 255.0
        
        axes[2, 1].imshow(blended_overlay)
        axes[2, 1].set_title('10. Final Red Overlay', fontweight='bold')
        axes[2, 1].axis('off')
        
        # Confidence overlay on original
        axes[2, 2].imshow(original_img)
        conf_overlay = axes[2, 2].imshow(confidence, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[2, 2].set_title('11. Confidence on Original', fontweight='bold')
        axes[2, 2].axis('off')
        
        # Processing statistics
        axes[2, 3].axis('off')
        stats_text = f"""Processing Statistics:
        
Original Size: {original_img.shape[:2]}
Max Confidence: {confidence.max():.3f}
Mean Confidence: {confidence.mean():.3f}
Confidence Threshold: 0.1

Base Prediction:
  - Max: {raw_output.max():.3f}
  - Mean: {raw_output.mean():.3f}
  - Pixels > {threshold}: {(raw_output > threshold).sum()}

Enhanced Prediction:
  - Max: {enhanced_output.max():.3f}
  - Mean: {enhanced_output.mean():.3f}
  - Pixels > {threshold}: {(enhanced_output > threshold).sum()}

Enhancement Gain:
  - Added Pixels: {((enhanced_output > threshold) & (raw_output <= threshold)).sum()}
  - Improvement: {(enhanced_output.mean() - raw_output.mean())*100:.1f}%"""
        
        if apply_dilation:
            dilation_stats = f"""

Dilation Applied:
  - Kernel Size: {dilate_kernel_size}x{dilate_kernel_size}
  - Iterations: {dilate_iterations}
  - Final Pixels: {(dilated_mask > 0.5).sum()}"""
            stats_text += dilation_stats
        
        axes[2, 3].text(0.05, 0.95, stats_text, transform=axes[2, 3].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[2, 3].set_title('12. Processing Statistics', fontweight='bold')
        
        plt.tight_layout()
        
        # Save pipeline visualization
        pipeline_save_path = os.path.join(pipeline_dir, f"{os.path.splitext(filename)[0]}_pipeline.png")
        plt.savefig(pipeline_save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save individual pipeline steps
        steps_dir = os.path.join(pipeline_dir, 'individual_steps', os.path.splitext(filename)[0])
        os.makedirs(steps_dir, exist_ok=True)
        
        # Save each step as individual image
        cv2.imwrite(os.path.join(steps_dir, '01_original.png'), 
                   cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(steps_dir, '02_grayscale.png'), 
                   (grayscale * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(steps_dir, '03_orientation_map.png'), 
                   cv2.cvtColor(orientation_color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(steps_dir, '04_confidence_map.png'), 
                   (confidence * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(steps_dir, '05_masked_orientation.png'), 
                   cv2.cvtColor(masked_orientation_color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(steps_dir, '06_base_prediction.png'), 
                   (raw_output * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(steps_dir, '07_enhanced_prediction.png'), 
                   (enhanced_output * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(steps_dir, '08_binary_mask.png'), 
                   (binary_mask * 255).astype(np.uint8))
        if apply_dilation:
            cv2.imwrite(os.path.join(steps_dir, '09_dilated_mask.png'), dilated_mask_uint8)
        cv2.imwrite(os.path.join(steps_dir, '10_final_overlay.png'), 
                   cv2.cvtColor((blended_overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        print(f"✓ Pipeline visualization saved: {pipeline_save_path}")
        print(f"✓ Individual steps saved: {steps_dir}")
        
        return pipeline_save_path, steps_dir

def main():
    parser = argparse.ArgumentParser(description='EXACT Mask Overlay with Complete Pipeline Visualization')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--images_dir', type=str, required=True, help='Directory with original images')
    parser.add_argument('--output_dir', type=str, default='predictions_pipeline', help='Output directory')
    parser.add_argument('--backbone', type=str, default='resnet101', 
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--decoder_channels', type=str, default='1024,512,256,128,64')
    parser.add_argument('--threshold', type=float, default=0.3, help='Threshold for binary mask')
    parser.add_argument('--target_size', type=str, default='256,144', help='Target image size for processing')
    parser.add_argument('--confidence_boost', type=float, default=0.4, help='Confidence boost factor')
    parser.add_argument('--overlay_alpha', type=float, default=0.6, help='Overlay transparency')
    parser.add_argument('--only_overlay', action='store_true', help='Only create overlays from existing masks')
    parser.add_argument('--visualize_pipeline', action='store_true', default=True, help='Create pipeline visualizations')
    parser.add_argument('--max_visualize', type=int, default=5, help='Maximum number of images to visualize pipeline')
    
    # New dilation arguments
    parser.add_argument('--apply_dilation', action='store_true', help='Apply morphological dilation to masks')
    parser.add_argument('--dilate_kernel_size', type=int, default=3, help='Kernel size for morphological dilation')
    parser.add_argument('--dilate_iterations', type=int, default=1, help='Number of dilation iterations')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    enhanced_masks_dir = os.path.join(args.output_dir, 'enhanced_masks')
    red_overlays_dir = os.path.join(args.output_dir, 'red_overlays')
    os.makedirs(enhanced_masks_dir, exist_ok=True)
    os.makedirs(red_overlays_dir, exist_ok=True)
    
    # Parse arguments
    target_size = tuple(map(int, args.target_size.split(',')))
    decoder_channels = [int(x) for x in args.decoder_channels.split(',')]
    
    print(f"\n{'='*70}")
    print(f"COMPLETE PIPELINE VISUALIZATION - Gabor + ResNet UNet")
    print(f"{'='*70}")
    print(f"Model: {args.model_path}")
    print(f"Images: {args.images_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Pipeline visualization: {'Enabled' if args.visualize_pipeline else 'Disabled'}")
    print(f"Max visualizations: {args.max_visualize}")
    print(f"Morphological dilation: {'Enabled' if args.apply_dilation else 'Disabled'}")
    if args.apply_dilation:
        print(f"Dilation kernel size: {args.dilate_kernel_size}x{args.dilate_kernel_size}")
        print(f"Dilation iterations: {args.dilate_iterations}")
    
    if not args.only_overlay:
        # STEP 1: Generate Enhanced Masks with Pipeline Visualization
        print(f"\n{'='*50}")
        print(f"STEP 1: Generating Enhanced Masks with Pipeline Visualization")
        print(f"{'='*50}")
        
        # Initialize model
        model = UNetWithResNetBackbone(
            in_channels=3, 
            out_channels=1, 
            backbone=args.backbone,
            decoder_channels=decoder_channels,
            use_gabor=True,
            pretrained=False
        ).to(device)
        
        # Load trained model
        print(f"Loading model...")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        
        # Create dataset
        dataset = ImageDataset(args.images_dir, target_size=target_size)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        
        print(f"Generating masks and visualizations for {len(dataset)} images...")
        
        visualized_count = 0
        with torch.no_grad():
            for inputs, filenames, original_sizes in tqdm(dataloader, desc="Processing with visualization"):
                inputs = inputs.to(device)
                filename = filenames[0]
                
                # Get enhanced mask
                enhanced_mask = model.get_enhanced_mask(inputs, args.confidence_boost)
                
                # Convert to binary mask
                binary_mask = (enhanced_mask[0, 0] > args.threshold).float().cpu().numpy()
                mask_uint8 = (binary_mask * 255).astype(np.uint8)
                
                # Apply morphological dilation if requested
                if args.apply_dilation:
                    mask_uint8 = dilate_mask(mask_uint8, args.dilate_kernel_size, args.dilate_iterations)
                
                # Save enhanced mask
                mask_path = os.path.join(enhanced_masks_dir, f"{os.path.splitext(filename)[0]}.png")
                cv2.imwrite(mask_path, mask_uint8)
                
                # Create pipeline visualization for first few images
                if args.visualize_pipeline and visualized_count < args.max_visualize:
                    try:
                        pipeline_path, steps_dir = visualize_complete_pipeline(
                            model, inputs, filename, args.output_dir, 
                            args.threshold, args.confidence_boost,
                            args.apply_dilation, args.dilate_kernel_size, args.dilate_iterations
                        )
                        visualized_count += 1
                    except Exception as e:
                        print(f"Warning: Failed to create pipeline visualization for {filename}: {e}")
        
        print(f"✓ Mask generation completed!")
        if args.visualize_pipeline:
            print(f"✓ Created {visualized_count} pipeline visualizations")
    
    # STEP 2: Create Exact Overlays from Saved Masks
    print(f"\n{'='*50}")
    print(f"STEP 2: Creating Exact Overlays from Saved Masks")
    print(f"{'='*50}")
    
    # Get all enhanced mask files
    mask_files = [f for f in os.listdir(enhanced_masks_dir) if f.endswith('.png')]
    print(f"Found {len(mask_files)} enhanced masks to overlay")
    
    if len(mask_files) == 0:
        print("ERROR: No enhanced masks found! Run without --only_overlay first.")
        return
    
    processed = 0
    for mask_file in tqdm(mask_files, desc="Creating exact overlays"):
        # Extract original image name
        original_name = mask_file.replace('.png', '')
        
        # Find original image with any extension
        original_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.tif', '.JPG', '.PNG', '.JPEG', '.TIF']:
            test_path = os.path.join(args.images_dir, original_name + ext)
            if os.path.exists(test_path):
                original_path = test_path
                break
        
        if original_path is None:
            print(f"Could not find original image for {mask_file}")
            continue
        
        # Paths
        mask_path = os.path.join(enhanced_masks_dir, mask_file)
        output_path = os.path.join(red_overlays_dir, f"{original_name}_exact_overlay.png")
        
        # Create exact overlay using saved mask
        overlay_result = create_exact_overlay_from_saved_mask(
            original_path, mask_path, target_size, alpha=args.overlay_alpha
        )
        
        if overlay_result is not None:
            # Save result
            cv2.imwrite(output_path, cv2.cvtColor(overlay_result, cv2.COLOR_RGB2BGR))
            processed += 1
        else:
            print(f"Failed to create overlay for {mask_file}")
    
    print(f"\n{'='*70}")
    print(f"🎉 COMPLETE PIPELINE PROCESSING FINISHED! 🎉")
    print(f"{'='*70}")
    print(f"📊 Results Summary:")
    print(f"  ✓ Enhanced masks: {len(mask_files)} images")
    print(f"  ✓ Red overlays: {processed} images")
    if args.visualize_pipeline:
        print(f"  ✓ Pipeline visualizations: {min(len(mask_files), args.max_visualize)} detailed breakdowns")
    if args.apply_dilation:
        print(f"  ✓ Morphological dilation applied: {args.dilate_kernel_size}x{args.dilate_kernel_size} kernel, {args.dilate_iterations} iterations")
    print(f"\n📁 Output Structure:")
    print(f"  📂 {enhanced_masks_dir}/")
    print(f"  📂 {red_overlays_dir}/")
    if args.visualize_pipeline:
        print(f"  📂 {args.output_dir}/pipeline_visualization/")
        print(f"    └── individual_steps/ (step-by-step images)")
    print(f"\n🔍 Pipeline Visualization Shows:")
    print(f"  1️⃣  Original Input → 2️⃣  Grayscale → 3️⃣  Gabor Orientation")
    print(f"  4️⃣  Confidence Map → 5️⃣  Masked Orientation → 6️⃣  Base Prediction")
    print(f"  7️⃣  Enhanced Mask → 8️⃣  Enhancement Diff → 9️⃣  Binary Result")
    print(f"  🔟 Final Overlay → 1️⃣1️⃣ Confidence Overlay → 1️⃣2️⃣ Statistics")
    if args.apply_dilation:
        print(f"  🎯 Morphological dilation included in pipeline steps")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

'''
# Example command to run the script with dilation
python predict.py \
    --model_path best_model.pth \
    --images_dir /mnt/zone/B/Sayak/testimg \
    --output_dir predictions_pipeline \
    --backbone resnet101 \
    --decoder_channels "1024,512,256,128,64" \
    --visualize_pipeline \
    --max_visualize 5 \
    --threshold 0.3 \
    --apply_dilation \
    --dilate_kernel_size 5 \
    --dilate_iterations 2
'''
