import os
import cv2
import numpy as np
from tqdm import tqdm
import random
import math
import shutil
from pathlib import Path


def create_directories():
    """Create necessary directories for the dataset"""
    source_dir = r'/mnt/zone/B/Sayak/images_monkey'
    dataset_dir = r'/mnt/zone/B/Sayak/dataset'
    masks_dir = r'/mnt/zone/B/Sayak/dataset/masks'
    cage_dir = r'/mnt/zone/B/Sayak/cages'
    
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(cage_dir, exist_ok=True)
    
    return source_dir, dataset_dir, masks_dir, cage_dir


def load_and_resize_images(directory, target_size=(512, 288)):
    """Load all images from a directory, resize them, and preserve alpha channels"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    images = []
    image_paths = []
    
    print(f"Loading and resizing images from {directory} to {target_size}...")
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            # Use IMREAD_UNCHANGED to preserve alpha channel
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                # If image is grayscale, convert to RGB
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                # If image has no alpha but is a PNG, add alpha channel
                elif img.shape[2] == 3 and filename.lower().endswith('.png'):
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
                    img[:, :, 3] = 255  # Set alpha to fully opaque
                
                # Resize the image to target size
                if len(img.shape) > 2 and img.shape[2] == 4:  # Image with alpha channel
                    # Resize preserving alpha channel
                    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                else:  # RGB image
                    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                
                images.append(resized_img)
                image_paths.append(file_path)
    
    print(f"Loaded and resized {len(images)} images from {directory}")
    return images, image_paths


def random_zoom(image, min_zoom=0.8, max_zoom=1.5):
    """Apply random zoom to an image (both zoom in and zoom out), preserving alpha channel if present"""
    zoom_factor = random.uniform(min_zoom, max_zoom)
    h, w = image.shape[:2]
    
    # Check if image has alpha channel
    has_alpha = len(image.shape) > 2 and image.shape[2] == 4
    
    # Calculate new dimensions
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    
    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    if zoom_factor > 1.0:  # Zoom in - crop center
        y_offset = (new_h - h) // 2
        x_offset = (new_w - w) // 2
        result = resized[y_offset:y_offset+h, x_offset:x_offset+w]
    else:  # Zoom out - pad with transparent/black
        result = np.zeros_like(image)
        y_offset = (h - new_h) // 2
        x_offset = (w - new_w) // 2
        
        if has_alpha:
            # For images with alpha, make background transparent
            result[:, :, 3] = 0  # Transparent background
        
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return result


def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """Adjust brightness and contrast of an image, preserving alpha channel"""
    if len(image.shape) > 2 and image.shape[2] == 4:  # Image with alpha channel
        # Apply brightness/contrast only to RGB channels
        result = image.copy()
        result[:, :, :3] = cv2.convertScaleAbs(image[:, :, :3], alpha=contrast, beta=brightness)
        return result
    else:
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)


def adjust_saturation(image, saturation_factor=1.0):
    """Adjust saturation of an image, preserving alpha channel"""
    if len(image.shape) > 2 and image.shape[2] >= 3:
        # Convert to HSV for saturation adjustment
        if image.shape[2] == 4:  # Image with alpha
            rgb_part = image[:, :, :3]
            alpha_part = image[:, :, 3]
            hsv = cv2.cvtColor(rgb_part, cv2.COLOR_BGR2HSV).astype(np.float32)
        else:  # RGB image
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Adjust saturation
        hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Convert back to BGR
        result_rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        if len(image.shape) > 2 and image.shape[2] == 4:
            # Recombine with alpha channel
            result = np.dstack((result_rgb, alpha_part))
            return result
        else:
            return result_rgb
    else:
        return image


def overlay_cage(background, cage, augment=True):
    """Overlay a cage image with alpha channel on a background image and create a mask"""
    # Get background dimensions
    bg_h, bg_w = background.shape[:2]
    
    # Check if cage has alpha channel
    has_alpha = len(cage.shape) > 2 and cage.shape[2] == 4
    
    if not has_alpha:
        print("Warning: Cage image doesn't have alpha channel. Creating a simple mask.")
        # Convert to grayscale and threshold to create a binary mask
        if len(cage.shape) == 3:  # Color image without alpha
            cage_gray = cv2.cvtColor(cage, cv2.COLOR_BGR2GRAY)
        else:  # Already grayscale
            cage_gray = cage
            
        # Add an alpha channel based on thresholding
        _, alpha_channel = cv2.threshold(cage_gray, 10, 255, cv2.THRESH_BINARY)
        cage_rgb = cage if len(cage.shape) == 3 else cv2.cvtColor(cage, cv2.COLOR_GRAY2BGR)
        cage = np.dstack((cage_rgb, alpha_channel))
    
    # Resize cage to match background dimensions exactly
    resized_cage = cv2.resize(cage, (bg_w, bg_h), interpolation=cv2.INTER_AREA)
    
    # Apply augmentations if requested
    if augment:
        # Apply random zoom (both in and out)
        resized_cage = random_zoom(resized_cage, min_zoom=0.8, max_zoom=1.5)
        
        # Apply brightness/contrast adjustments
        brightness = random.randint(-30, 30)
        contrast = random.uniform(0.8, 1.3)
        resized_cage = adjust_brightness_contrast(resized_cage, brightness, contrast)
        
        # Apply saturation adjustment
        saturation = random.uniform(0.7, 1.4)
        resized_cage = adjust_saturation(resized_cage, saturation)
    
    # Extract alpha channel from the cage image
    alpha_channel = resized_cage[:, :, 3] / 255.0  # Normalize to 0-1
    
    # Extract RGB channels from the cage
    cage_rgb = resized_cage[:, :, :3]
    
    # Create a full-size mask based on the alpha channel
    full_mask = (alpha_channel * 255).astype(np.uint8)
    
    # Alpha blending: result = background * (1.0 - alpha) + cage * alpha
    blended = background * (1.0 - alpha_channel[:, :, np.newaxis]) + cage_rgb * alpha_channel[:, :, np.newaxis]
    
    return blended, full_mask


def generate_dataset(source_dir, dataset_dir, masks_dir, cage_dir, target_size=(512, 288)):
    """Generate the dataset by overlaying cages on source images"""
    # Load and resize source images and cage images
    source_images, source_paths = load_and_resize_images(source_dir, target_size)
    cage_images, cage_paths = load_and_resize_images(cage_dir, target_size)
    
    if not source_images or not cage_images:
        print("Error: No images found in source or cage directories.")
        return
    
    # Create a list to store generated image paths
    generated_images = []
    
    # Generate all permutations
    total_combinations = len(source_images) * len(cage_images)
    print(f"Generating {total_combinations} images with size {target_size}...")
    
    with tqdm(total=total_combinations) as pbar:
        for i, source_img in enumerate(source_images):
            source_name = os.path.basename(source_paths[i])
            source_name = os.path.splitext(source_name)[0]
            
            for j, cage_img in enumerate(cage_images):
                cage_name = os.path.basename(cage_paths[j])
                cage_name = os.path.splitext(cage_name)[0]
                
                # Generate a unique name for this combination
                output_name = f"{source_name}_cage_{cage_name}"
                
                # Create variations with augmentation
                overlayed_img, mask = overlay_cage(source_img, cage_img, augment=True)
                
                # Save the overlayed image
                output_path = os.path.join(dataset_dir, f"{output_name}.jpg")
                cv2.imwrite(output_path, overlayed_img)
                
                # Save the mask
                mask_path = os.path.join(masks_dir, f"{output_name}.jpg")
                cv2.imwrite(mask_path, mask)
                
                # Add to the list of generated images
                generated_images.append(output_path)
                
                pbar.update(1)
    
    print(f"Dataset generation complete. Generated {len(generated_images)} images.")
    return generated_images


def create_additional_augmentations(dataset_dir, masks_dir, num_augmentations=3, target_size=(512, 288)):
    """Create additional augmentations of existing dataset images"""
    # Get all generated images
    image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg') and 
                  os.path.isfile(os.path.join(dataset_dir, f)) and
                  os.path.isfile(os.path.join(masks_dir, f))]
    
    print(f"Creating {num_augmentations} additional augmentations for each of {len(image_files)} images...")
    
    with tqdm(total=len(image_files) * num_augmentations) as pbar:
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            
            # Load the image and its mask
            img_path = os.path.join(dataset_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file)
            
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None or mask is None:
                continue
            
            # Ensure images are at the target size
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            
            # Create augmentations
            for aug_idx in range(num_augmentations):
                # Apply color augmentations to the image
                aug_img = img.copy()
                
                # Apply brightness/contrast adjustments
                brightness = random.randint(-20, 20)
                contrast = random.uniform(0.9, 1.2)
                aug_img = adjust_brightness_contrast(aug_img, brightness, contrast)
                
                # Apply saturation adjustment
                saturation = random.uniform(0.8, 1.3)
                aug_img = adjust_saturation(aug_img, saturation)
                
                # Save augmented image and original mask
                aug_img_path = os.path.join(dataset_dir, f"{base_name}_aug{aug_idx}.jpg")
                aug_mask_path = os.path.join(masks_dir, f"{base_name}_aug{aug_idx}.jpg")
                
                cv2.imwrite(aug_img_path, aug_img)
                cv2.imwrite(aug_mask_path, mask)  # Keep original mask
                
                pbar.update(1)
    
    print("Additional augmentations complete.")


def main():
    # Define target size
    target_size = (512, 288)
    
    # Create directories
    source_dir, dataset_dir, masks_dir, cage_dir = create_directories()
    
    # Check if cage images need to be copied from another location
    cage_source = r'/media/big_zone/Sayak/cage_images'  # Adjust this to your actual cage images location
    if os.path.exists(cage_source) and os.path.isdir(cage_source) and len(os.listdir(cage_dir)) == 0:
        print(f"Copying cage images from {cage_source} to {cage_dir}...")
        for file in os.listdir(cage_source):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                shutil.copy(os.path.join(cage_source, file), os.path.join(cage_dir, file))
    
    # Generate the dataset with specified target size
    generated_images = generate_dataset(source_dir, dataset_dir, masks_dir, cage_dir, target_size)
    
    # Create additional augmentations
    create_additional_augmentations(dataset_dir, masks_dir, num_augmentations=3, target_size=target_size)
    
    print("Dataset creation complete!")
    
    # Print statistics
    total_images = len([f for f in os.listdir(dataset_dir) if f.endswith('.jpg')])
    total_masks = len([f for f in os.listdir(masks_dir) if f.endswith('.jpg')])
    
    print(f"Total images in dataset: {total_images}")
    print(f"Total masks in dataset: {total_masks}")
    print(f"All images are resized to {target_size[0]}x{target_size[1]} pixels")


if __name__ == "__main__":
    main()
