#trainer.py

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import argparse
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import random_split
import subprocess
import math
from PIL import Image
import torchvision.transforms as transforms
from skimage.filters import difference_of_gaussians
from torch.utils.tensorboard import SummaryWriter
import torchvision
import pdb
import time
import io
import torch.utils.checkpoint

# Configure PyTorch memory allocation to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

def get_free_gpu():
    """Returns the index of the GPU with the most available memory."""
    try:
        # Using nvidia-smi to get memory usage
        gpu_stats = subprocess.check_output(
            ["nvidia-smi", "--format=csv", "--query-gpu=index,memory.used,memory.total"],
            universal_newlines=True
        )
        gpu_lines = gpu_stats.strip().split('\n')[1:]  # Skip header
        # Parse memory info
        memory_available = []
        for line in gpu_lines:
            values = line.split(',')
            index = int(values[0])
            used_mem = int(values[1].strip().split()[0])  # Used memory in MiB
            total_mem = int(values[2].strip().split()[0])  # Total memory in MiB
            free_mem = total_mem - used_mem
            memory_available.append((index, free_mem))
        
        # Print GPU memory information
        print(f"\n{'='*70}")
        print(f"GPU Memory Information:")
        for idx, free_mem in memory_available:
            total_mem = free_mem + int(values[1].strip().split()[0])
            print(f"  - GPU {idx}: {free_mem/1024:.2f} GB free / {total_mem/1024:.2f} GB total ({free_mem/total_mem*100:.1f}% available)")
        print(f"{'='*70}\n")
        
        # Return GPU with most free memory
        selected_gpu = max(memory_available, key=lambda x: x[1])[0]
        print(f"Selected GPU {selected_gpu} with most available memory")
        return int(selected_gpu)
    except Exception as e:
        print(f"Error getting GPU memory info: {e}")
        return 0  # Default to GPU 0 if there's an error

# Check for GPU availability and select the one with most free memory
if torch.cuda.is_available():
    # If CUDA_VISIBLE_DEVICES is set, respect it
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        device = torch.device("cuda:0")  # Use the first (and only) visible GPU
        print(f"Using GPU 0 as specified by CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        free_gpu_id = get_free_gpu()
        device = torch.device(f"cuda:{free_gpu_id}")
        torch.cuda.set_device(device)
        print(f"Using GPU {free_gpu_id} with {torch.cuda.get_device_name(free_gpu_id)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Gabor Filter Module with Learnable Parameters
class calOrientationGabor(nn.Module):
    """
    Gabor filter module for extracting orientation features from images.
    
    Gabor filters are linear filters used for texture analysis. They detect
    frequency content in specific directions in an image. This implementation
    creates a bank of Gabor filters at different orientations to extract
    directional features, which are particularly useful for detecting bars
    and grid patterns.
    
    The module applies multiple Gabor filters at different orientations and
    computes a confidence map based on the filter responses. This helps in
    identifying the dominant orientation at each pixel.
    """
    def __init__(self, channel_in=1, channel_out=1, stride=1):
        super(calOrientationGabor, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.Tensor = torch.cuda.FloatTensor

        # Reduced number of orientation kernels to save memory
        self.numKernels = 72 # Reduced from 180 to 36 (still provides good orientation resolution)
        
        # Confidence thresholds for filtering weak responses
        self.clamp_confidence_low = 0.0
        self.clamp_confidence_high = 0.2
        
        # Make Gabor parameters learnable
        self.sigma_x = nn.Parameter(torch.ones(1) * 1.8)
        self.sigma_y = nn.Parameter(torch.ones(1) * 2.4)
        self.Lambda = nn.Parameter(torch.ones(1) * 4.0)
        self.kernel_size = 17  # Keep kernel size fixed

    def filter(self, image, label, threshold, variance_data, orient_data, max_resp_data):
        """
        Apply Gabor filters at different orientations to extract directional features.
        
        Args:
            image: Input image tensor
            label: Label tensor (used as a placeholder)
            threshold: Threshold for filtering weak responses
            variance_data, orient_data, max_resp_data: Previous filter data for iterative refinement
            
        Returns:
            confidenceTensor: Confidence map of orientation detection
            variance_data: Updated variance data
            orient_data: Updated orientation data
        """
        # Process orientations in batches to save memory
        batch_size = 6
        H, W = image.size()[2:4]
        
        # Initialize tensor to store all responses
        all_responses = []
        all_orients = []
        
        # Process orientations in batches
        for batch_start in range(0, self.numKernels, batch_size):
            batch_end = min(batch_start + batch_size, self.numKernels)
            batch_resArray = []
            batch_orients = []
            
            for iOrient in range(batch_start, batch_end):
                # Create Gabor filter at current orientation with learnable parameters
                theta = nn.Parameter(torch.ones(self.channel_out) * (math.pi * iOrient / self.numKernels)).to(device)
                GaborKernel = self.gabor_fn(self.kernel_size, self.channel_in, self.channel_out, theta, 
                                            self.sigma_x, self.sigma_y, self.Lambda)
                
                # Apply filter and store response
                response = F.conv2d(image, GaborKernel, padding=self.kernel_size//2)
                batch_resArray.append(response)
                
                # Create orientation tensor for this orientation
                orient_tensor = torch.ones(1, 1, H, W).to(device) * math.pi * iOrient / self.numKernels
                batch_orients.append(orient_tensor)
            
            # Concatenate batch responses and orientations
            all_responses.extend(batch_resArray)
            all_orients.extend(batch_orients)
            
            # Clear GPU cache after each batch
            torch.cuda.empty_cache()
        
        # Concatenate all responses and orientations
        resTensor = torch.cat(all_responses, dim=1)
        orient = torch.cat(all_orients, dim=1)
        
        # Find the maximum response at each pixel (strongest orientation)
        resTensor = torch.abs(resTensor)
        max_resp = torch.max(resTensor, dim=1, keepdim=True)[0]
        maxResTensor = torch.argmax(resTensor, dim=1, keepdim=True).float()
        best_orientTensor = maxResTensor * math.pi / self.numKernels

        # Calculate orientation difference
        orient_diff = torch.minimum(torch.abs(best_orientTensor - orient),
                           torch.minimum(torch.abs(best_orientTensor - orient - math.pi),
                                      torch.abs(best_orientTensor - orient + math.pi)))

        # Compute confidence based on response variance
        resp_diff = resTensor - max_resp
        variance = torch.sum(orient_diff * resp_diff * resp_diff, dim=1, keepdim=True)
        variance = variance ** (1 / 2)
        
        # Update orientation data where variance is higher
        orient_data = torch.where(variance > variance_data, best_orientTensor, orient_data)
        max_resp_data = torch.where(variance > variance_data, max_resp, max_resp_data)
        variance_data = torch.where(variance > variance_data, variance, variance_data)

        # Normalize response and variance
        max_all_resp = torch.max(max_resp_data)
        max_all_var = torch.max(variance_data)
        
        # Avoid division by zero
        if max_all_resp > 0:
            max_resp_data = max_resp_data / max_all_resp
        if max_all_var > 0:
            variance_data = variance_data / max_all_var

        # Calculate final confidence tensor
        confidenceTensor = (variance_data - self.clamp_confidence_low) / (
                        self.clamp_confidence_high - self.clamp_confidence_low)
        confidenceTensor = confidenceTensor.clamp(0, 1)

        # Clear intermediate tensors to save memory
        del resTensor, orient, max_resp, maxResTensor, best_orientTensor, orient_diff, resp_diff, variance
        torch.cuda.empty_cache()

        return confidenceTensor, variance_data, orient_data

    def forward(self, image, label, iter=1, threshold=0.0):
        """
        Extract orientation features from the input image.
        
        Args:
            image: Input grayscale image tensor
            label: Label tensor (used as a placeholder)
            iter: Number of iterations for refinement
            threshold: Threshold for filtering weak responses
            
        Returns:
            orientTwoChannel: Two-channel tensor with sin and cos of orientation
            best_orientTensor: Tensor containing the best orientation at each pixel
            confidenceTensor: Confidence map of orientation detection
        """
        # Initialize data tensors
        H, W = image.size()[2:4]
        variance_data = torch.ones(1, 1, H, W).to(device) * 0
        orient_data = torch.ones(1, 1, H, W).to(device) * 0
        max_resp_data = torch.ones(1, 1, H, W).to(device) * 0

        # Iteratively refine orientation detection
        for i in range(iter):
            confidenceTensor, variance_data, orient_data = self.filter(
                image, label, threshold, variance_data, orient_data, max_resp_data
            )
            image = confidenceTensor
        
        # Apply threshold to confidence tensor
        confidenceTensor[confidenceTensor < threshold] = 0
        best_orientTensor = orient_data
        
        # Convert orientation to two-channel representation (sin/cos)
        # This makes it easier for the network to learn orientation features
        orientTwoChannel = torch.cat([torch.sin(best_orientTensor), torch.cos(best_orientTensor)], dim=1)
        return orientTwoChannel, best_orientTensor, confidenceTensor

    def gabor_fn(self, kernel_size, channel_in, channel_out, theta, sigma_x, sigma_y, Lambda, phase=0.):
        """
        Create Gabor filter kernels with learnable parameters.
        
        A Gabor filter is a linear filter used for edge detection. It's defined by a sinusoidal
        wave multiplied by a Gaussian function. The parameters control the properties of the filter:
        - theta: Orientation of the filter
        - sigma_x, sigma_y: Standard deviations of the Gaussian envelope (learnable)
        - Lambda: Wavelength of the sinusoidal factor (learnable)
        - phase: Phase offset of the sinusoidal factor
        
        Args:
            kernel_size: Size of the Gabor kernel
            channel_in: Number of input channels
            channel_out: Number of output channels
            theta: Orientation of the filter
            sigma_x, sigma_y: Standard deviations of the Gaussian envelope (learnable)
            Lambda: Wavelength of the sinusoidal factor (learnable)
            phase: Phase offset of the sinusoidal factor
            
        Returns:
            gb: Gabor filter kernel
        """
        # Use learnable parameters
        sigma_x_expanded = sigma_x.expand(channel_out)
        sigma_y_expanded = sigma_y.expand(channel_out)
        Lambda_expanded = Lambda.expand(channel_out)
        psi = nn.Parameter(torch.ones(channel_out) * phase).to(device)

        # Create coordinate grid for the kernel
        xmax = kernel_size // 2
        ymax = kernel_size // 2
        xmin = -xmax
        ymin = -ymax
        ksize = xmax - xmin + 1
        y_0 = torch.arange(ymin, ymax + 1).to(device).float() - 0.5
        y = y_0.view(1, -1).repeat(channel_out, channel_in, ksize, 1).float()
        x_0 = torch.arange(xmin, xmax + 1).to(device).float() - 0.5
        x = x_0.view(-1, 1).repeat(channel_out, channel_in, 1, ksize).float()  

        # Apply rotation to coordinates
        x_theta = x * torch.cos(theta.view(-1, 1, 1, 1)) + y * torch.sin(theta.view(-1, 1, 1, 1))
        y_theta = -x * torch.sin(theta.view(-1, 1, 1, 1)) + y * torch.cos(theta.view(-1, 1, 1, 1))

        # Create Gabor filter: Gaussian envelope * cosine wave
        gb = torch.exp(
            -.5 * (x_theta ** 2 / sigma_x_expanded.view(-1, 1, 1, 1) ** 2 + 
                   y_theta ** 2 / sigma_y_expanded.view(-1, 1, 1, 1) ** 2)) \
             * torch.cos(2 * math.pi * x_theta / Lambda_expanded.view(-1, 1, 1, 1) + psi.view(-1, 1, 1, 1))

        return gb

# Helper functions for Gabor orientation
def normalize_tensor(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    return x / torch.maximum(norm, torch.ones_like(norm) * 1e-8)

def convert_numpy(tensor):
    tensor = torch.squeeze(tensor, 0)
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.data.cpu().numpy()
    return tensor

# Define the U-Net model with Gabor orientation features
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

def get_ori_map(orient):
    orient = orient[...,None] 
    cos = np.cos(orient)
    orient_img = np.concatenate([cos * (cos >= 0), np.sin(orient), np.abs(cos) * (cos < 0)], axis=-1)
    orient_img = (orient_img * 255).round().astype('uint8')
    
    return orient_img

class UNetWithGabor(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], use_gabor=True):
        super(UNetWithGabor, self).__init__()
        self.use_gabor = use_gabor
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_checkpoint = True  # Enable gradient checkpointing to save memory
        
        # Gabor orientation filter - adds 2 channels (sin/cos of orientation)
        if self.use_gabor:
            self.gabor_filter = calOrientationGabor(channel_in=1, channel_out=1)
            # Update in_channels to add 2 more channels for orientation features
            in_channels += 2
        
        # Down part of UNet
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Extract the grayscale image for Gabor filtering
        if self.use_gabor:
            # Convert RGB to grayscale to feed into the Gabor filter
            gray = torch.mean(x, dim=1, keepdim=True)
            
            # Create a dummy label tensor (all ones)
            dummy_label = torch.ones_like(gray)
            
            # Apply Gabor filter to get orientation features
            orientTwoChannel, best_ori, confidence = self.gabor_filter(gray, dummy_label)
            
            # Create a masked orientation map where low-confidence areas are set to zero
            confidence_threshold = 0.2
            bconf = best_ori * (confidence > confidence_threshold)

            # Concatenate the masked orientation features with input image
            # This provides the network with orientation information only in high-confidence areas
            bconfTwoChannel = torch.cat([torch.sin(bconf), torch.cos(bconf)], dim=1)
            x = torch.cat([x, bconfTwoChannel], dim=1)
            
            # Clear intermediate tensors to save memory
            del gray, dummy_label, orientTwoChannel, best_ori, confidence, bconf, bconfTwoChannel
            torch.cuda.empty_cache()
        
        skip_connections = []

        # Use gradient checkpointing for the down path to save memory
        if self.use_checkpoint:
            for i, down in enumerate(self.downs):
                # Fixed: Added use_reentrant=False to checkpoint
                x = torch.utils.checkpoint.checkpoint(down, x, use_reentrant=False)
                skip_connections.append(x)
                x = self.pool(x)
        else:
            for down in self.downs:
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Use gradient checkpointing for the up path
        for idx in range(0, len(self.ups), 2):
            if self.use_checkpoint:
                # Fixed: Added use_reentrant=False to checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ups[idx], x, use_reentrant=False)
            else:
                x = self.ups[idx](x)
                
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True
                )
            concat_skip = torch.cat((skip_connection, x), dim=1)
            
            if self.use_checkpoint:
                # Fixed: Added use_reentrant=False to checkpoint
                x = torch.utils.checkpoint.checkpoint(self.ups[idx+1], concat_skip, use_reentrant=False)
            else:
                x = self.ups[idx+1](concat_skip)

        x = self.final_conv(x)
        return x

# Custom Dataset class to load image-mask pairs from separate folders
class ImageMaskDataset(Dataset):
    def __init__(self, images_dir, masks_dir, target_size=(256, 144), use_gabor=True):  # Reduced image size
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.target_size = target_size
        self.use_gabor = use_gabor
        
        # Get all image files
        self.images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif'))]
        
        print(f"\n{'='*70}")
        print(f"Creating Image-Mask Dataset:")
        print(f"  - Images directory: {images_dir}")
        print(f"  - Masks directory: {masks_dir}")
        print(f"  - Target size: {target_size}")
        print(f"  - Total images found: {len(self.images)}")
        if self.use_gabor:
            print(f"  - Using Gabor orientation features")
        print(f"{'='*70}\n")
        
        # Verify that corresponding masks exist
        missing_masks = []
        for img_file in self.images:
            mask_file = img_file  # Assuming same filename for corresponding mask
            if not os.path.exists(os.path.join(masks_dir, mask_file)):
                missing_masks.append(mask_file)
        
        if missing_masks:
            print(f"WARNING: {len(missing_masks)} masks are missing for corresponding images.")
            print(f"First few missing masks: {missing_masks[:5]}")
            # Filter out images without masks
            self.images = [img for img in self.images if img not in missing_masks]
            print(f"Filtered dataset contains {len(self.images)} valid image-mask pairs")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get image and mask filenames
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error loading image: {img_path}")
            # Return a blank image and mask as fallback
            blank = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            return torch.from_numpy(blank.transpose(2, 0, 1)).float(), torch.zeros(1, self.target_size[1], self.target_size[0]).float()
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error loading mask: {mask_path}")
            # Return the image with a blank mask as fallback
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Resize image and mask to target size
        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size)
        
        # Normalize
        image = image / 255.0
        mask = mask / 255.0
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        
        return image_tensor, mask_tensor

# Enhanced function to count model parameters with detailed breakdown
def count_parameters(model):
    """Count the total number of trainable parameters in the model with detailed breakdown."""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate parameters by module type
    module_params = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only count leaf modules
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                module_type = module.__class__.__name__
                if module_type not in module_params:
                    module_params[module_type] = 0
                module_params[module_type] += params
    
    # Calculate Gabor filter parameters separately
    gabor_params = 0
    if hasattr(model, 'use_gabor') and model.use_gabor:
        gabor_params = sum(p.numel() for p in model.gabor_filter.parameters() if p.requires_grad)
    
    return total_params, module_params, gabor_params

# Calculate IoU metric
def calculate_iou(pred, target, threshold=0.5):
    # Apply sigmoid to get probabilities since we're using BCEWithLogitsLoss
    pred = torch.sigmoid(pred)
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    if union < 1e-6:
        return 1.0  # If both pred and target are empty, IoU is 1
    
    return intersection / union

# Function to log Gabor visualizations for a batch of images
def log_gabor_visualizations(writer, model, inputs, targets, global_step, prefix="training"):
    """Log Gabor filter visualizations for a batch of images"""
    # Convert RGB to grayscale
    gray = torch.mean(inputs, dim=1, keepdim=True)
    dummy_label = torch.ones_like(gray)
    
    # Apply Gabor filter
    with torch.no_grad():  # Use no_grad to save memory
        orientTwoChannel, best_ori, confidence = model.gabor_filter(gray, dummy_label)
        
        # Create a masked orientation map
        confidence_threshold = 0.2
        bconf = best_ori * (confidence > confidence_threshold)
        
        # Log original images
        img_grid = torchvision.utils.make_grid(inputs, normalize=True, scale_each=True)
        writer.add_image(f'{prefix}/original_images', img_grid, global_step)
        
        # Log ground truth masks
        mask_grid = torchvision.utils.make_grid(targets, normalize=True, scale_each=True)
        writer.add_image(f'{prefix}/ground_truth_masks', mask_grid, global_step)
        
        # Log orientation channels
        orient_grid = torchvision.utils.make_grid(orientTwoChannel, normalize=True, scale_each=True)
        writer.add_image(f'{prefix}/gabor/orientation_channels', orient_grid, global_step)
        
        # Log confidence map
        conf_grid = torchvision.utils.make_grid(confidence, normalize=True, scale_each=True)
        writer.add_image(f'{prefix}/gabor/confidence_map', conf_grid, global_step)
        
        # Log masked orientation map for just one image to save memory
        if inputs.size(0) > 0:
            # Convert orientation to color image for visualization
            bconf_np = bconf[0][0].cpu().numpy()
            ori_map = get_ori_map(bconf_np)
            
            # Handle different dimension cases
            if len(ori_map.shape) == 3:
                # If ori_map has 3 dimensions (height, width, channels)
                ori_map_tensor = torch.from_numpy(ori_map.transpose(2, 0, 1) / 255.0).float()
            else:
                # If ori_map has 2 dimensions (height, width)
                ori_map_tensor = torch.from_numpy(ori_map / 255.0).float().unsqueeze(0)
            
            writer.add_image(f'{prefix}/gabor/orientation_map_0', ori_map_tensor, global_step)
            
            # Side-by-side comparison
            input_img = inputs[0].cpu()
            
            # Ensure ori_map_tensor has 3 channels for proper concatenation
            if ori_map_tensor.size(0) < 3:
                ori_map_tensor = ori_map_tensor.repeat(3, 1, 1)
            
            comparison = torch.cat([input_img, ori_map_tensor], dim=2)
            writer.add_image(f'{prefix}/gabor/comparison_0', comparison, global_step)
    
    # Clear memory
    del gray, dummy_label, orientTwoChannel, best_ori, confidence, bconf
    torch.cuda.empty_cache()

# Log Gabor filter parameters to TensorBoard
def log_gabor_parameters(writer, model, epoch):
    """Log Gabor filter parameters to TensorBoard"""
    if hasattr(model, 'use_gabor') and model.use_gabor:
        writer.add_scalar('Gabor/sigma_x', model.gabor_filter.sigma_x.item(), epoch)
        writer.add_scalar('Gabor/sigma_y', model.gabor_filter.sigma_y.item(), epoch)
        writer.add_scalar('Gabor/Lambda', model.gabor_filter.Lambda.item(), epoch)

# Log Gabor filter parameter gradients
def log_gabor_gradients(writer, model, global_step):
    """Log Gabor filter parameter gradients after backward pass"""
    if hasattr(model, 'use_gabor') and model.use_gabor:
        if model.gabor_filter.sigma_x.grad is not None:
            writer.add_scalar('Gradients/sigma_x', model.gabor_filter.sigma_x.grad.norm().item(), global_step)
        if model.gabor_filter.sigma_y.grad is not None:
            writer.add_scalar('Gradients/sigma_y', model.gabor_filter.sigma_y.grad.norm().item(), global_step)
        if model.gabor_filter.Lambda.grad is not None:
            writer.add_scalar('Gradients/Lambda', model.gabor_filter.Lambda.grad.norm().item(), global_step)

def train_model(model, train_dataset, val_dataset, criterion, optimizer, batch_size=4, num_epochs=25, save_path='best_model.pth', verbose=1, log_file=None):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_ious = []
    
    # Use torch.amp for mixed precision training
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()
    
    # Initialize TensorBoard writer
    log_dir = os.path.join(os.path.dirname(log_file), 'tensorboard_logs')
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    # Global step counter for TensorBoard
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*70}")
        
        # Clear cache at the beginning of each epoch
        torch.cuda.empty_cache()
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=2, pin_memory=True)  # Reduced num_workers
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=2, pin_memory=True)  # Reduced num_workers
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        if verbose == 1:
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        else:
            train_iterator = train_loader
        
        for batch_idx, (inputs, targets) in enumerate(train_iterator):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision - FIXED: use torch.amp.autocast('cuda')
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Log gradients for Gabor parameters
            log_gabor_gradients(writer, model, global_step)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
            
            # Log batch loss to TensorBoard
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            
            # Log images and Gabor visualizations less frequently to save memory
            if batch_idx % 50 == 0 and model.use_gabor:  # Log every 50th batch
                log_gabor_visualizations(writer, model, inputs, targets, global_step, prefix="training")
                
                # Also log model predictions
                with torch.no_grad():
                    # Apply sigmoid to get probabilities
                    probs = torch.sigmoid(outputs)
                    pred_grid = torchvision.utils.make_grid(probs, normalize=True, scale_each=True)
                    writer.add_image('training/predictions', pred_grid, global_step)
            
            if verbose == 2 or (batch_idx % 50 == 0 and verbose == 1):
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Increment global step
            global_step += 1
            
            # Clear cache periodically
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Log epoch loss to TensorBoard
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        
        # Log Gabor parameters at the end of each epoch
        log_gabor_parameters(writer, model, epoch)
        
        # Clear cache before validation
        torch.cuda.empty_cache()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        print(f"\nStarting validation for Epoch {epoch+1}/{num_epochs}")
        
        if verbose == 1:
            val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        else:
            val_iterator = val_loader
        
        # For visualization in TensorBoard
        val_step = 0
        
        # Create a grid of validation predictions for TensorBoard
        val_inputs_to_log = []
        val_targets_to_log = []
        val_preds_to_log = []
        val_ious_to_log = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_iterator):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Use mixed precision for validation too - FIXED: use torch.amp.autocast('cuda')
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                
                # Calculate IoU for each image in batch
                batch_iou = 0.0
                for i in range(inputs.size(0)):
                    img_iou = calculate_iou(outputs[i], targets[i])
                    batch_iou += img_iou
                    
                    # Store IoU for each image in TensorBoard
                    writer.add_scalar('IoU/val_individual', img_iou, global_step + i)
                    
                    # Store a few examples for visualization
                    if len(val_inputs_to_log) < 8 and batch_idx % 10 == 0:  # Reduced from 16 to 8 examples
                        val_inputs_to_log.append(inputs[i].cpu())
                        val_targets_to_log.append(targets[i].cpu())
                        val_preds_to_log.append(torch.sigmoid(outputs[i]).cpu())
                        val_ious_to_log.append(img_iou)
                
                val_iou += batch_iou
                
                # Log validation images less frequently
                if batch_idx % 10 == 0 and model.use_gabor and batch_idx < 30:  # Log every 10th batch, max 3 batches
                    # Apply sigmoid to get probabilities
                    probs = torch.sigmoid(outputs)
                    
                    # Log Gabor visualizations
                    log_gabor_visualizations(writer, model, inputs, targets, val_step, prefix="validation")
                    
                    # Log predictions
                    pred_grid = torchvision.utils.make_grid(probs, normalize=True, scale_each=True)
                    writer.add_image('validation/predictions', pred_grid, val_step)
                
                if verbose == 2 or (batch_idx % 20 == 0 and verbose == 1):
                    print(f"  Val Batch {batch_idx+1}/{len(val_loader)}, Loss: {loss.item():.4f}, IoU: {batch_iou/inputs.size(0):.4f}")
                
                val_step += 1
                
                # Clear cache periodically
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
        
        # Create a grid with validation predictions and IoU scores
        if val_inputs_to_log:
            try:
                # Create a figure with subplots for each validation example
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # Reduced from 4x4 to 2x4
                axes = axes.flatten()
                
                for i, (input_img, target, pred, iou) in enumerate(zip(val_inputs_to_log, val_targets_to_log, val_preds_to_log, val_ious_to_log)):
                    if i >= 8:  # Limit to 8 examples
                        break
                        
                    # Convert tensors to numpy arrays
                    input_np = input_img.permute(1, 2, 0).numpy()
                    target_np = target.squeeze().numpy()
                    pred_np = pred.squeeze().numpy()
                    
                    # Display input image
                    axes[i].imshow(input_np)
                    axes[i].set_title(f"IoU: {iou:.4f}")
                    axes[i].axis('off')
                    
                    # Overlay target and prediction
                    target_rgba = np.zeros((*target_np.shape, 4))
                    target_rgba[..., 1] = target_np  # Green channel for target
                    target_rgba[..., 3] = target_np * 0.5  # Alpha channel
                    
                    pred_rgba = np.zeros((*pred_np.shape, 4))
                    pred_rgba[..., 0] = pred_np  # Red channel for prediction
                    pred_rgba[..., 3] = pred_np * 0.5  # Alpha channel
                    
                    axes[i].imshow(target_rgba)
                    axes[i].imshow(pred_rgba)
                
                # Save figure to TensorBoard
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                val_pred_img = Image.open(buf)
                val_pred_tensor = transforms.ToTensor()(val_pred_img)
                writer.add_image('validation/predictions_with_iou', val_pred_tensor, epoch)
                plt.close(fig)
            except Exception as e:
                print(f"Error creating validation visualization: {e}")
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        val_iou = val_iou / len(val_loader.dataset)
        val_ious.append(val_iou)
        
        # Log validation metrics to TensorBoard
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('IoU/validation', val_iou, epoch)
        
        # Print epoch results
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{num_epochs} Results:")
        print(f"  - Train Loss: {train_loss:.4f}")
        print(f"  - Val Loss: {val_loss:.4f}")
        print(f"  - Val IoU: {val_iou:.4f}")
        print(f"{'='*70}")
        
        # Log results
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}\n")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  - Model saved with validation loss: {val_loss:.4f} (improved)")
            if log_file:
                with open(log_file, 'a') as f:
                    f.write(f"Model saved at epoch {epoch+1} with validation loss: {val_loss:.4f}\n")
        else:
            print(f"  - No improvement in validation loss")
        
        # Clear cache after each epoch
        torch.cuda.empty_cache()
    
    # Close TensorBoard writer
    writer.close()
    
    # Load best model
    print(f"\nLoading best model from {save_path}")
    model.load_state_dict(torch.load(save_path))
    
    return model, train_losses, val_losses, val_ious

# Function to visualize predictions
def predict_on_validation_set(model, val_dataset, batch_size=4, num_samples=5, log_file=None):
    model.eval()
    
    print(f"\n{'='*70}")
    print(f"Generating Validation Predictions")
    print(f"{'='*70}")
    
    # Create a directory for saving validation images if it doesn't exist
    save_dir = os.path.join(os.path.dirname(log_file), 'validation_predictions')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving validation predictions to: {save_dir}")
    
    # Create a validation loader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize TensorBoard writer for prediction visualization
    log_dir = os.path.join(os.path.dirname(log_file), 'tensorboard_logs')
    writer = SummaryWriter(log_dir)
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if i >= num_samples:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Use mixed precision - FIXED: use torch.amp.autocast('cuda')
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
            
            # Apply sigmoid to get probabilities
            outputs = torch.sigmoid(outputs)
            
            # Calculate IoU for each image
            batch_ious = []
            for j in range(inputs.size(0)):
                iou = calculate_iou(outputs[j], targets[j])
                batch_ious.append(iou.item())
            
            # Convert tensors to numpy arrays for visualization
            inputs_np = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            
            print(f"Processing validation batch {i+1}/{num_samples}")
            
            # Plot and save results
            for j in range(min(inputs.size(0), 4)):  # Show up to 4 images from the batch
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(np.transpose(inputs_np[j], (1, 2, 0)))
                axes[0].set_title('Input Image')
                axes[0].axis('off')
                
                # Ground truth mask
                axes[1].imshow(targets_np[j, 0], cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                # Predicted mask
                axes[2].imshow(outputs_np[j, 0], cmap='gray')
                axes[2].set_title(f'Prediction (IoU: {batch_ious[j]:.4f})')
                axes[2].axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(save_dir, f'val_pred_{i}_{j}.png')
                plt.savefig(save_path)
                
                # Also save to TensorBoard
                writer.add_figure(f'val_predictions/batch_{i}_img_{j}', fig, global_step=i*batch_size+j)
                
                plt.close()
                
                print(f"  - Saved prediction {j+1} to {save_path}")
                
                if log_file:
                    with open(log_file, 'a') as f:
                        f.write(f"Saved validation prediction: val_pred_{i}_{j}.png (IoU: {batch_ious[j]:.4f})\n")
            
            # Clear cache after each batch
            torch.cuda.empty_cache()
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"Validation prediction generation complete.")

# Function to run TensorBoard
def run_tensorboard(log_dir):
    """
    Start TensorBoard server in a separate process
    """
    import subprocess
    import webbrowser
    import time
    
    print(f"\n{'='*70}")
    print(f"Starting TensorBoard")
    print(f"{'='*70}")
    print(f"TensorBoard logs directory: {log_dir}")
    
    # Start TensorBoard server with increased samples_per_plugin for images
    tensorboard_process = subprocess.Popen(
        ["tensorboard", "--logdir", log_dir, "--port", "6006", "--samples_per_plugin", "images=1000"],  # Reduced from 2000 to 1000
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give it a moment to start
    time.sleep(3)
    
    # Open browser
    webbrowser.open("http://localhost:6006")
    
    print("TensorBoard started at http://localhost:6006")
    print("Press Ctrl+C in this terminal to stop TensorBoard when you're done")
    
    return tensorboard_process

# Main function
def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='U-Net with Gabor Training for Image Segmentation')
    
    # Dataset paths
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directory containing the original images')
    parser.add_argument('--masks_dir', type=str, required=True,
                        help='Directory containing the ground truth masks')
    
    # Training hyperparameters
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--verbose', type=int, default=1, choices=[0, 1, 2],
                        help='Verbosity mode: 0=silent, 1=progress bar, 2=one line per batch')
    
    # Model hyperparameters
    parser.add_argument('--in_channels', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--out_channels', type=int, default=1,
                        help='Number of output channels')
    parser.add_argument('--features', type=str, default='32,64,128,256,512',
                        help='Feature dimensions for each layer, comma-separated')
    parser.add_argument('--use_gabor', action='store_true',
                        help='Use Gabor orientation features')
    
    # Output settings
    parser.add_argument('--model_path', type=str, default='best_model.pth',
                        help='Path to save the best model')
    parser.add_argument('--log_file', type=str, default='log.txt',
                        help='Path to save training logs')
    
    # TensorBoard settings
    parser.add_argument('--run_tensorboard', action='store_true',
                        help='Start TensorBoard server automatically')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set use_gabor to True by default
    if not args.use_gabor:
        print("Note: --use_gabor flag not provided, but using Gabor features by default")
        args.use_gabor = True
    
    print(f"\n{'='*70}")
    print(f"U-Net with Gabor Training for Image Segmentation")
    print(f"{'='*70}")
    print(f"Command line arguments:")
    for arg, value in vars(args).items():
        print(f"  - {arg}: {value}")
    print(f"{'='*70}\n")
    
    # Create log file with
    # Create log file with header
    with open(args.log_file, 'w') as f:
        f.write(f"U-Net with Gabor Training for Image Segmentation - Log File\n")
        f.write(f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command line arguments: {vars(args)}\n")
        f.write(f"Using GPU: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'}\n")
        f.write("-" * 80 + "\n\n")
    
    # Convert features string to list of integers
    features = [int(x) for x in args.features.split(',')]
    
    # Initialize model with parsed arguments
    print(f"Creating U-Net with Gabor model with features: {features}")
    model = UNetWithGabor(
        in_channels=args.in_channels, 
        out_channels=args.out_channels, 
        features=features,
        use_gabor=args.use_gabor
    ).to(device)
    
    # Enhanced parameter counting with detailed breakdown
    total_params, module_params, gabor_params = count_parameters(model)
    
    print(f"\n{'='*70}")
    print(f"Model Parameter Information:")
    print(f"  - Total trainable parameters: {total_params:,}")
    if gabor_params > 0:
        print(f"  - Gabor filter parameters: {gabor_params:,} ({gabor_params/total_params*100:.2f}%)")
    print(f"  - Parameters by module type:")
    for module_type, params in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
        print(f"    - {module_type}: {params:,} ({params/total_params*100:.2f}%)")
    print(f"{'='*70}\n")
    
    # Also log to file
    with open(args.log_file, 'a') as f:
        f.write(f"Model Parameter Information:\n")
        f.write(f"  - Total trainable parameters: {total_params:,}\n")
        if gabor_params > 0:
            f.write(f"  - Gabor filter parameters: {gabor_params:,} ({gabor_params/total_params*100:.2f}%)\n")
        f.write(f"  - Parameters by module type:\n")
        for module_type, params in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
            f.write(f"    - {module_type}: {params:,} ({params/total_params*100:.2f}%)\n")
        f.write("\n")
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)  # Using AdamW for better memory efficiency
    print(f"Using AdamW optimizer with learning rate: {args.learning_rate}")
    print(f"Using BCEWithLogitsLoss for training")
    
    # Create dataset with image-mask pairs
    dataset = ImageMaskDataset(args.images_dir, args.masks_dir, target_size=(256, 144), use_gabor=args.use_gabor)  # Reduced image size
    
    # Split dataset into train and validation
    dataset_size = len(dataset)
    train_size = int((1 - args.val_split) * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\n{'='*70}")
    print(f"Dataset Split Information:")
    print(f"  - Total image-mask pairs: {dataset_size}")
    print(f"  - Training samples: {train_size} ({train_size/dataset_size*100:.1f}%)")
    print(f"  - Validation samples: {val_size} ({val_size/dataset_size*100:.1f}%)")
    print(f"{'='*70}\n")
    
    # Set up TensorBoard directory
    log_dir = os.path.join(os.path.dirname(args.log_file), 'tensorboard_logs')
    
    # Start TensorBoard if requested
    tensorboard_process = None
    if args.run_tensorboard:
        tensorboard_process = run_tensorboard(log_dir)
    
    # Train the model
    print(f"\n{'='*70}")
    print(f"Starting Training")
    print(f"{'='*70}")
    
    trained_model, train_losses, val_losses, val_ious = train_model(
        model, train_dataset, val_dataset, criterion, optimizer, 
        batch_size=args.batch, num_epochs=args.num_epochs, save_path=args.model_path, 
        verbose=args.verbose, log_file=args.log_file
    )
    
    # Visualize predictions on validation set
    print("\nGenerating validation predictions...")
    predict_on_validation_set(model, val_dataset, batch_size=args.batch, num_samples=5, log_file=args.log_file)
    
    print(f"\n{'='*70}")
    print(f"Training completed! Best model saved to {args.model_path}")
    print(f"TensorBoard logs saved to {log_dir}")
    print(f"To view TensorBoard logs, run: tensorboard --logdir={log_dir}")
    print(f"{'='*70}")
    
    with open(args.log_file, 'a') as f:
        f.write(f"Training completed! Best model saved to {args.model_path}\n")
        f.write(f"TensorBoard logs saved to {log_dir}\n")
    
    # Keep TensorBoard running until user interrupts
    if tensorboard_process:
        try:
            print("\nTensorBoard is running. Press Ctrl+C to stop and exit.")
            tensorboard_process.wait()
        except KeyboardInterrupt:
            print("\nStopping TensorBoard and exiting...")
            tensorboard_process.terminate()

if __name__ == "__main__":
    main()
