#!/usr/bin/env python3
"""
infer.py - ViTPose inference script with interactive bbox drawing GUI


Usage:
CUDA_VISIBLE_DEVICES=1 python infer.py --cfg configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_large_ap10k_256x192.py --checkpoint checkpoints/ap10k.pth --image "/mnt/zone/B/Sayak/organized/1deer/clip3/1deer_clip3_(9)10_cage5.png" --out-dir ./ICVGIP_results --debug --confidence 0.3


OR for batch processing:
python infer.py --cfg configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_large_ap10k_256x192.py --checkpoint checkpoints/ap10k.pth --img-dir /mnt/zone/B/Sayak/crfill/results --out-dir /mnt/zone/B/Sayak/crfill/kankaria_results --debug --confidence 0.3
"""
import argparse
import os
import mmcv
import torch
import numpy as np
import cv2
import json
import base64
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
    def __init__(self, image_path):
        self.image_path = image_path
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
                cv2.rectangle(temp_img, (self.ix, self.iy), (x, y), (0, 255, 0), 3)
                cv2.imshow('Draw Bounding Box', temp_img)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Finalize rectangle
            cv2.rectangle(self.image, (self.ix, self.iy), (x, y), (0, 255, 0), 3)
            self.bbox = (min(self.ix, x), min(self.iy, y), 
                        max(self.ix, x), max(self.iy, y))
            cv2.imshow('Draw Bounding Box', self.image)
    
    def get_bbox(self):
        """Interactive bbox drawing interface"""
        global shutdown_requested
        
        print("\n=== Interactive Bounding Box Drawing ===")
        print("Instructions:")
        print("1. Click and drag to draw a bounding box around the animal")
        print("2. Press 'r' to reset and redraw")
        print("3. Press 'ENTER' to confirm the bounding box")
        print("4. Press 'q' to quit without processing")
        print("5. Press 'Ctrl+C' for graceful exit")
        
        cv2.imshow('Draw Bounding Box', self.image)
        
        try:
            while not shutdown_requested:
                key = cv2.waitKey(100) & 0xFF  # Check more frequently for Ctrl+C
                
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
                    
                elif key == ord('q'):  # Quit
                    print("Exiting...")
                    cv2.destroyAllWindows()
                    return None
                    
        except KeyboardInterrupt:
            print("\n GUI interrupted by user")
            cv2.destroyAllWindows()
            return None
        
        cv2.destroyAllWindows()
        
        if shutdown_requested:
            print(" Shutdown requested during bbox drawing")
            return None
            
        return self.bbox


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, help='Config file')
    parser.add_argument('--checkpoint', required=True, help='Model weights')
    
    # Make these mutually exclusive
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', help='Single image path (enables GUI mode)')
    group.add_argument('--img-dir', help='Folder with input images (batch mode)')
    
    parser.add_argument('--out-dir', default='pose_results', help='Save directory')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--confidence', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    return parser.parse_args()


def get_ap10k_keypoint_names():
    """Get AP10K keypoint names in order"""
    return [
        "L_Eye", "R_Eye", "Nose", "Neck", "Root_of_tail",
        "L_Shoulder", "L_Elbow", "L_F_Paw", "R_Shoulder", "R_Elbow", "R_F_Paw",
        "L_Hip", "L_Knee", "L_B_Paw", "R_Hip", "R_Knee", "R_B_Paw"
    ]


def encode_image_to_base64(image_path):
    """Encode image to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except:
        return None


def save_keypoints_to_json(keypoints, bbox, image_path, output_path, confidence_threshold=0.3):
    """Save keypoints in LabelMe-style JSON format similar to AP36K"""
    
    keypoint_names = get_ap10k_keypoint_names()
    
    # Create shapes list
    shapes = []
    
    # Add bounding box as rectangle
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        bbox_shape = {
            "label": "animal_bbox",
            "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
            "group_id": 1,
            "shape_type": "rectangle",
            "flags": {}
        }
        shapes.append(bbox_shape)
    
    # Add individual keypoints as points
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > confidence_threshold and i < len(keypoint_names):
            keypoint_shape = {
                "label": keypoint_names[i],
                "points": [[float(x), float(y)]],
                "group_id": 1,
                "shape_type": "point",
                "flags": {}
            }
            shapes.append(keypoint_shape)
    
    # Get image dimensions
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    
    # Create the JSON structure
    json_data = {
        "version": "4.6.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": encode_image_to_base64(image_path),
        "imageHeight": height,
        "imageWidth": width
    }
    
    # Save JSON file
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    return json_data


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
            
            cv2.line(img, start_point, end_point, color, 6)
    
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


def process_single_image_with_gui(args, model):
    """Process single image with interactive bbox drawing"""
    global shutdown_requested
    
    if shutdown_requested:
        print(" Shutdown requested, skipping processing")
        return False
        
    print(f"Loading image: {args.image}")
    
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return False
    
    try:
        # Initialize bbox drawer
        drawer = BBoxDrawer(args.image)
        bbox = drawer.get_bbox()
        
        if bbox is None or shutdown_requested:
            print("No bounding box drawn or shutdown requested. Exiting...")
            return False
        
        x1, y1, x2, y2 = bbox
        print(f"Processing with bbox: ({x1}, {y1}, {x2}, {y2})")
        
        # Check for shutdown before inference
        if shutdown_requested:
            print(" Shutdown requested before inference")
            return False
        
        # Create detection result with user-drawn bbox
        det_result = [{'bbox': np.array([x1, y1, x2, y2, 1.0])}]
        
        # Perform inference
        result = inference_top_down_pose_model(
            model, args.image, det_result,
            format='xyxy', 
            dataset=model.cfg.data['test']['type'],
            return_heatmap=False, 
            outputs=None)
        
        if args.debug:
            print(f"Debug - Result type: {type(result)}")
            print(f"Debug - Result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        
        # Check for shutdown before processing results
        if shutdown_requested:
            print(" Shutdown requested during processing")
            return False
        
        # Handle tuple format: (predictions, heatmap)
        if isinstance(result, tuple) and len(result) >= 1:
            predictions = result[0]  # Extract predictions from tuple
            
            # Extract keypoints from predictions
            keypoints = None
            if isinstance(predictions, list) and len(predictions) > 0:
                if isinstance(predictions[0], dict) and 'keypoints' in predictions[0]:
                    keypoints = predictions[0]['keypoints']
                elif isinstance(predictions[0], (list, np.ndarray)):
                    keypoints = predictions[0]
                else:
                    keypoints = predictions
                    
            if keypoints is not None:
                # Convert to numpy array if needed
                if isinstance(keypoints, torch.Tensor):
                    keypoints = keypoints.cpu().numpy()
                
                keypoints = np.array(keypoints)
                if keypoints.ndim == 1:
                    keypoints = keypoints.reshape(-1, 3)
                
                if keypoints.shape[1] == 2:
                    keypoints = np.hstack([keypoints, np.ones((keypoints.shape[0], 1))])
                
                # Load original image and create visualization
                img = cv2.imread(args.image)
                
                # Draw the bounding box on the image
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                
                # Draw pose
                img_vis = draw_ap10k_pose(img, keypoints, args.confidence)
                
                # Check for shutdown before saving
                if shutdown_requested:
                    print(" Shutdown requested before saving results")
                    return False
                
                # Save results
                mkdir_or_exist(args.out_dir)
                fname = os.path.basename(args.image)
                vis_path = os.path.join(args.out_dir, fname)
                cv2.imwrite(vis_path, img_vis)
                
                # Save raw results (PyTorch format)
                result_path = vis_path.rsplit('.', 1)[0] + '.pt'
                torch.save({'result': result, 'bbox': bbox}, result_path)
                
                # Save keypoints in JSON format
                json_path = vis_path.rsplit('.', 1)[0] + '.json'
                save_keypoints_to_json(keypoints, bbox, args.image, json_path, args.confidence)
                
                # Count confident keypoints
                confident_kpts = sum(1 for kpt in keypoints if kpt[2] > args.confidence)
                print(f"✓ Detected {confident_kpts}/{len(keypoints)} confident keypoints")
                print(f"📁 Results saved to:")
                print(f"   - Visualization: {vis_path}")
                print(f"   - PyTorch data: {result_path}")
                print(f"   - JSON keypoints: {json_path}")
                
                # Show result with interruption handling
                if not shutdown_requested:
                    try:
                        cv2.namedWindow('Pose Estimation Result', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Pose Estimation Result', 1000, 700)
                        cv2.imshow('Pose Estimation Result', img_vis)
                        print("\nPress any key to close the result window...")
                        
                        while not shutdown_requested:
                            key = cv2.waitKey(100) & 0xFF
                            if key != 255:  # Any key pressed
                                break
                        
                        cv2.destroyAllWindows()
                    except KeyboardInterrupt:
                        cv2.destroyAllWindows()
                        print("\n Result display interrupted")
                
                return True
            else:
                print("⚠ No valid keypoints found")
                return False
        else:
            print("⚠ Unexpected result format")
            return False
            
    except KeyboardInterrupt:
        print("\n Processing interrupted by user")
        cv2.destroyAllWindows()
        return False
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return False


def process_batch_images(args, model):
    """Process multiple images from directory (original functionality)"""
    global shutdown_requested
    
    image_files = [f for f in sorted(os.listdir(args.img_dir)) 
                   if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))]
    
    print(f"Found {len(image_files)} images to process")
    
    successful_count = 0
    failed_count = 0


    try:
        for fname in image_files:
            if shutdown_requested:
                print(f"\n ⚠️  Shutdown requested. Stopping batch processing.")
                print(f" Processed {successful_count}/{len(image_files)} images before interruption")
                break
                
            try:
                img_path = os.path.join(args.img_dir, fname)
                print(f"\nProcessing: {fname}...")
                
                # Load image
                img = mmcv.imread(img_path)
                if img is None:
                    print(f"  ❌ Could not load image: {fname}")
                    failed_count += 1
                    continue
                    
                h, w = img.shape[:2]
                print(f"  Image dimensions: {w}x{h}")
                
                # Create bounding box for entire image
                det_result = [{'bbox': np.array([0, 0, w, h, 1.0])}]
                
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
                
                # Process result (same logic as before)
                if isinstance(result, tuple) and len(result) >= 1:
                    predictions = result[0]
                    
                    if isinstance(predictions, list) and len(predictions) > 0:
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
                            
                            # Create visualization
                            img_vis = img.copy()
                            img_vis = draw_ap10k_pose(img_vis, keypoints, args.confidence)
                            
                            # Check for shutdown before saving
                            if shutdown_requested:
                                break
                            
                            # Save results
                            vis_path = os.path.join(args.out_dir, fname)
                            cv2.imwrite(vis_path, img_vis)
                            
                            # Save PyTorch format
                            torch.save(result, vis_path.rsplit('.', 1)[0] + '.pt')
                            
                            # Save JSON format
                            json_path = vis_path.rsplit('.', 1)[0] + '.json'
                            bbox_full = (0, 0, w, h)  # Full image bbox for batch processing
                            save_keypoints_to_json(keypoints, bbox_full, img_path, json_path, args.confidence)
                            
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
    print(f"❌ Failed/No poses: {failed_count} images")
    print(f"📁 Results saved to: {args.out_dir}")
    
    if shutdown_requested:
        print(f" Process terminated early due to user request")


def main():
    global shutdown_requested
    
    try:
        args = parse_args()
        mkdir_or_exist(args.out_dir)


        print(f"Loading model from: {args.checkpoint}")
        model = init_pose_model(args.cfg, args.checkpoint, device=args.device)
        
        if shutdown_requested:
            print(" Shutdown requested during model loading")
            return
        
        if args.image:
            # Single image mode with GUI
            print("\n=== Single Image Mode with Interactive GUI ===")
            success = process_single_image_with_gui(args, model)
            if success and not shutdown_requested:
                print("✓ Processing completed successfully!")
            elif shutdown_requested:
                print(" 🛑 Processing interrupted by user")
            else:
                print("❌ Processing failed!")
        else:
            # Batch processing mode
            print("\n=== Batch Processing Mode ===")
            process_batch_images(args, model)
            
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
            print(" ✅ exiting my precious code completed")
        print("👋 Goodbye!")


if __name__ == '__main__':
    main()
