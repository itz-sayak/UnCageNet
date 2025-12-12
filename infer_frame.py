#!/usr/bin/env python3
"""
infer_frame.py - ViTPose batch inference script with interactive bbox drawing (images only)

Usage:
CUDA_VISIBLE_DEVICES=1 python infer_frame.py --cfg configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_large_ap10k_256x192.py --checkpoint checkpoints/ap10k.pth --img-dir /mnt/zone/B/Sayak/crfill/results --out-dir /mnt/zone/B/Sayak/crfill/kankaria_results --debug --confidence 0.3
"""
import argparse
import os
import mmcv
import torch
import numpy as np
import cv2
import signal
import sys
from mmcv import mkdir_or_exist
from mmpose.apis import (init_pose_model, inference_top_down_pose_model)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    """Handle Ctrl+C (SIGINT) gracefully"""
    global shutdown_requested
    print('\n\n ⚠️  Ctrl+C received! Initiating graceful shutdown...')
    print(' Please wait for current operation to complete...')
    shutdown_requested = True
    
    # If we're in OpenCV window context, destroy all windows
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    # Exit immediately if pressed twice
    if hasattr(signal_handler, 'called'):
        print('\n 🛑 Force exit requested!')
        sys.exit(1)
    signal_handler.called = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

class BBoxDrawer:
    def __init__(self, image_path, image_name):
        self.image_path = image_path
        self.image_name = image_name
        self.image = cv2.imread(image_path)
        self.clone = self.image.copy()
        self.bbox = None
        self.drawing = False
        self.ix = -1
        self.iy = -1
        
        # Create window and set mouse callback
        cv2.namedWindow('Draw Bounding Box', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Draw Bounding Box', 1000, 700)
        cv2.setMouseCallback('Draw Bounding Box', self.draw_rectangle)
        
    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Show live rectangle while dragging
                temp_img = self.clone.copy()
                cv2.rectangle(temp_img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('Draw Bounding Box', temp_img)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Finalize rectangle
            cv2.rectangle(self.image, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            self.bbox = (min(self.ix, x), min(self.iy, y), 
                        max(self.ix, x), max(self.iy, y))
            cv2.imshow('Draw Bounding Box', self.image)
    
    def get_bbox(self):
        """Interactive bbox drawing interface"""
        global shutdown_requested
        
        print(f"\n=== Drawing bbox for: {self.image_name} ===")
        print("Instructions:")
        print("1. Click and drag to draw a bounding box around the animal")
        print("2. Press 'r' to reset and redraw")
        print("3. Press 'ENTER' to confirm and process")
        print("4. Press 's' to skip this image")
        print("5. Press 'q' to quit batch processing")
        print("6. Press 'Ctrl+C' for graceful exit")
        
        cv2.imshow('Draw Bounding Box', self.image)
        
        try:
            while not shutdown_requested:
                key = cv2.waitKey(100) & 0xFF
                
                if key == 13:  # Enter key
                    if self.bbox is not None:
                        print(f"✓ Bounding box confirmed: {self.bbox}")
                        break
                    else:
                        print("⚠ Please draw a bounding box first!")
                        
                elif key == ord('r'):  # Reset
                    print("Resetting bounding box...")
                    self.image = self.clone.copy()
                    self.bbox = None
                    cv2.imshow('Draw Bounding Box', self.image)
                    
                elif key == ord('s'):  # Skip
                    print(f"Skipping {self.image_name}...")
                    cv2.destroyAllWindows()
                    return "skip"
                    
                elif key == ord('q'):  # Quit batch
                    print("Quitting batch processing...")
                    cv2.destroyAllWindows()
                    return "quit"
                    
        except KeyboardInterrupt:
            print(f"\n GUI interrupted by user for {self.image_name}")
            cv2.destroyAllWindows()
            return None
        
        cv2.destroyAllWindows()
        
        if shutdown_requested:
            print(f" Shutdown requested during bbox drawing for {self.image_name}")
            return None
            
        return self.bbox

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='Config file')
    parser.add_argument('--checkpoint', required=True, help='Model weights')
    parser.add_argument('--img-dir', required=True, help='Folder with input images')
    parser.add_argument('--out-dir', required=True, help='Save directory')
    parser.add_argument('--device', default='cuda:0', help='Device to use')
    parser.add_argument('--confidence', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    return parser.parse_args()

def draw_ap10k_pose(img, keypoints, confidence_threshold=0.3):
    """Draw AP10K animal pose with proper skeleton connections"""
    
    # AP10K skeleton connections based on dataset info
    skeleton_connections = [
        (0, 1),   # L_Eye - R_Eye
        (0, 2),   # L_Eye - Nose  
        (1, 2),   # R_Eye - Nose
        (2, 3),   # Nose - Neck
        (3, 4),   # Neck - Root of tail
        (3, 5),   # Neck - L_Shoulder
        (5, 6),   # L_Shoulder - L_Elbow
        (6, 7),   # L_Elbow - L_F_Paw
        (3, 8),   # Neck - R_Shoulder
        (8, 9),   # R_Shoulder - R_Elbow
        (9, 10),  # R_Elbow - R_F_Paw
        (4, 11),  # Root of tail - L_Hip
        (11, 12), # L_Hip - L_Knee
        (12, 13), # L_Knee - L_B_Paw
        (4, 14),  # Root of tail - R_Hip
        (14, 15), # R_Hip - R_Knee
        (15, 16), # R_Knee - R_B_Paw
    ]
    
    # Colors for different body parts
    colors = {
        'head': (0, 255, 0),      # Green for head
        'body': (255, 0, 0),      # Blue for body
        'left_front': (0, 255, 255),   # Yellow for left front
        'right_front': (255, 0, 255),  # Magenta for right front
        'left_back': (255, 255, 0),    # Cyan for left back
        'right_back': (128, 0, 128),   # Purple for right back
    }
    
    # Draw skeleton connections
    for start_idx, end_idx in skeleton_connections:
        if (start_idx < len(keypoints) and end_idx < len(keypoints) and
            keypoints[start_idx][2] > confidence_threshold and 
            keypoints[end_idx][2] > confidence_threshold):
            
            start_point = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            end_point = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
            
            # Choose color based on body part
            if start_idx <= 4 or end_idx <= 4:
                color = colors['body']
            elif start_idx in [5, 6, 7] or end_idx in [5, 6, 7]:
                color = colors['left_front']
            elif start_idx in [8, 9, 10] or end_idx in [8, 9, 10]:
                color = colors['right_front']
            elif start_idx in [11, 12, 13] or end_idx in [11, 12, 13]:
                color = colors['left_back']
            else:
                color = colors['right_back']
            
            cv2.line(img, start_point, end_point, color, 2)
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold:
            center = (int(x), int(y))
            
            # Choose color based on keypoint type
            if i <= 2:  # Head keypoints
                color = colors['head']
            elif i <= 4:  # Body keypoints
                color = colors['body']
            elif i <= 7:  # Left front leg
                color = colors['left_front']
            elif i <= 10:  # Right front leg
                color = colors['right_front']
            elif i <= 13:  # Left back leg
                color = colors['left_back']
            else:  # Right back leg
                color = colors['right_back']
            
            # Draw keypoint
            cv2.circle(img, center, 4, color, -1)
            cv2.circle(img, center, 4, (255, 255, 255), 1)
            
            # Draw keypoint label
            cv2.putText(img, f"{i}", (int(x) + 6, int(y)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return img

def process_batch_images_with_bbox(args, model):
    """Process multiple images from directory with interactive bbox drawing"""
    global shutdown_requested
    
    # Get all image files
    image_files = [f for f in sorted(os.listdir(args.img_dir)) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))]
    
    print(f"Found {len(image_files)} images to process")
    print(f"Output directory: {args.out_dir}")
    print(f"Confidence threshold: {args.confidence}")
    
    successful_count = 0
    failed_count = 0
    skipped_count = 0

    try:
        for i, fname in enumerate(image_files, 1):
            if shutdown_requested:
                print(f"\n ⚠️  Shutdown requested. Stopping batch processing.")
                print(f" Processed {successful_count}/{len(image_files)} images before interruption")
                break
                
            try:
                img_path = os.path.join(args.img_dir, fname)
                print(f"\n[{i}/{len(image_files)}] Loading: {fname}...")
                
                # Load image
                img = mmcv.imread(img_path)
                if img is None:
                    print(f"  ❌ Could not load image: {fname}")
                    failed_count += 1
                    continue
                    
                h, w = img.shape[:2]
                if args.debug:
                    print(f"  Image dimensions: {w}x{h}")
                
                # Interactive bbox drawing
                drawer = BBoxDrawer(img_path, fname)
                bbox = drawer.get_bbox()
                
                # Handle different return values
                if bbox == "skip":
                    print(f"  ⏭️  Skipped: {fname}")
                    skipped_count += 1
                    continue
                elif bbox == "quit":
                    print(f"  🛑 Batch processing stopped by user at {fname}")
                    break
                elif bbox is None or shutdown_requested:
                    print(f"  ⚠️  No bbox drawn or shutdown requested for {fname}")
                    break
                
                x1, y1, x2, y2 = bbox
                print(f"  Processing with bbox: ({x1}, {y1}, {x2}, {y2})")
                
                # Create detection result with user-drawn bbox
                det_result = [{'bbox': np.array([x1, y1, x2, y2, 1.0])}]
                
                # Check for shutdown before inference
                if shutdown_requested:
                    break
                
                # Perform inference
                result = inference_top_down_pose_model(
                    model, img_path, det_result,
                    format='xyxy', 
                    dataset=model.cfg.data['test']['type'],
                    return_heatmap=False, 
                    outputs=None)
                
                if args.debug:
                    print(f"  Debug - Result type: {type(result)}")
                    print(f"  Debug - Result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                
                # Process result
                if isinstance(result, tuple) and len(result) >= 1:
                    predictions = result[0]
                    
                    if isinstance(predictions, list) and len(predictions) > 0:
                        # Extract keypoints
                        if isinstance(predictions[0], dict) and 'keypoints' in predictions[0]:
                            keypoints = predictions[0]['keypoints']
                        elif isinstance(predictions[0], (list, np.ndarray)):
                            keypoints = predictions[0]
                        else:
                            keypoints = predictions
                            
                        if isinstance(keypoints, torch.Tensor):
                            keypoints = keypoints.cpu().numpy()
                        
                        if isinstance(keypoints, (list, np.ndarray)) and len(keypoints) > 0:
                            keypoints = np.array(keypoints)
                            if keypoints.ndim == 1:
                                keypoints = keypoints.reshape(-1, 3)
                            
                            if keypoints.shape[1] == 2:
                                keypoints = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
                            
                            # Check for shutdown before saving
                            if shutdown_requested:
                                break
                            
                            # Load original image and create visualization
                            img_original = cv2.imread(img_path)
                            
                            # Draw the bounding box on the image
                            cv2.rectangle(img_original, (x1, y1), (x2, y2), (255, 255, 255), 2)
                            cv2.putText(img_original, 'Detection Area', (x1, y1-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Create visualization with pose
                            img_vis = draw_ap10k_pose(img_original, keypoints, args.confidence)
                            
                            # Save only the visualization image
                            vis_path = os.path.join(args.out_dir, fname)
                            cv2.imwrite(vis_path, img_vis)
                            
                            # Count confident keypoints
                            confident_kpts = sum(1 for kpt in keypoints if kpt[2] > args.confidence)
                            print(f"  ✓ {fname}: {confident_kpts}/{len(keypoints)} confident keypoints detected")
                            successful_count += 1
                        else:
                            print(f"  ⚠ {fname}: No valid keypoints found")
                            failed_count += 1
                    else:
                        print(f"  ⚠ {fname}: Empty predictions")
                        failed_count += 1
                else:
                    print(f"  ⚠ {fname}: Unexpected result format")
                    failed_count += 1
                    
            except Exception as e:
                print(f"  ❌ {fname}: Error - {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                failed_count += 1
                
    except KeyboardInterrupt:
        print(f"\n Batch processing interrupted by user")
    
    print(f"\n=== Batch Processing Complete ===")
    print(f"✓ Successfully processed: {successful_count} images")
    print(f"⏭️  Skipped: {skipped_count} images")
    print(f"❌ Failed/No poses: {failed_count} images")
    print(f"📁 Results saved to: {args.out_dir}")
    
    if shutdown_requested:
        print(f" Process terminated early due to user request")

def main():
    global shutdown_requested
    
    try:
        args = parse_args()
        
        # Create output directory
        mkdir_or_exist(args.out_dir)
        print(f"Created output directory: {args.out_dir}")

        # Load model
        print(f"Loading model from: {args.checkpoint}")
        print(f"Using device: {args.device}")
        model = init_pose_model(args.cfg, args.checkpoint, device=args.device)
        
        if shutdown_requested:
            print(" Shutdown requested during model loading")
            return
        
        print(f"Model loaded successfully!")
        
        # Start batch processing with interactive bbox drawing
        print("\n=== Interactive Batch Processing Mode (Images Only) ===")
        process_batch_images_with_bbox(args, model)
            
    except KeyboardInterrupt:
        print("\n 🛑 Main process interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up OpenCV windows
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        if shutdown_requested:
            print(" ✅ Graceful shutdown completed")
        print("👋 Goodbye!")

if __name__ == '__main__':
    main()
