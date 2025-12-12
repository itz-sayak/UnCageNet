#!/bin/bash
#image.sh

rm -rf /mnt/zone/B/Sayak/predictions_image
rm -rf /mnt/zone/B/Sayak/crfill/results/
rm -rf /mnt/zone/B/Sayak/crfill/results_image
rm -rf /mnt/zone/B/Sayak/ViTPose/pose_results_interactive/
rm -rf /mnt/zone/B/Sayak/ViTPose/bboxes_interactive/
rm -rf /mnt/zone/B/Sayak/ViTPose/pose_results_interactive/*_keypoints.json


# Exit immediately if a command exits with a non-zero status
set -e

# Initialize conda for shell use
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "⚠️ Could not find conda.sh. Please adjust the path to your conda installation."
    exit 1
fi

# Step 1: Mask Prediction using deepfill environment         #    --model_path best_model.pth \
echo "🔍 Starting Mask Prediction (deepfill env)..."
conda activate deepfill
CUDA_VISIBLE_DEVICES=1 python /mnt/zone/B/Sayak/predict.py \
    --model_path /mnt/zone/B/Sayak/model.pth \
    --images_dir /mnt/zone/B/Sayak/output \
    --output_dir predictions_image \
    --backbone resnet101 \
    --decoder_channels "1024,512,256,128,64" \
    --visualize_pipeline \
    --max_visualize 5 \
    --confidence_boost 0.23 \
    --threshold 0.2 \
    --apply_dilation \
    --dilate_kernel_size 2 \
    --dilate_iterations 1

# Step 2: Inpainting as CR-Fill using deepfill environment
echo "🧹 Switching to deepfill env for Inpainting and Pose Estimation..."
conda activate deepfill

echo "🧹 Starting Inpainting with CR-Fill..."
pushd /mnt/zone/B/Sayak/crfill > /dev/null
CUDA_VISIBLE_DEVICES=1 python test.py \
    --batchSize 1 \
    --nThreads 1 \
    --name objrmv \
    --dataset_mode testimage \
    --image_dir /mnt/zone/B/Sayak/output \
    --mask_dir /mnt/zone/B/Sayak/predictions_image/enhanced_masks \
    --output_dir ./results \
    --model inpaint \
    --netG baseconv \
    --which_epoch latest \
    --load_baseg
popd > /dev/null

# Step 3: Interactive Pose Estimation with ViTPose using GUI bounding box selection
echo "🧍 Starting Interactive Pose Estimation with STEP..."
echo "📝 You will be able to draw bounding boxes for each image interactively"

pushd /mnt/zone/B/IJCV_Code > /dev/null
onda activate myenv

bash mnt/zone/B/IJCV_Code/run.sh

# # Check if results directory exists and contains images
# RESULTS_DIR="/mnt/zone/B/Sayak/crfill/results"
# if [ ! -d "$RESULTS_DIR" ]; then
#     echo "❌ Error: Results directory not found: $RESULTS_DIR"
#     exit 1
# fi

# # Count images in the results directory
# IMAGE_COUNT=$(find "$RESULTS_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.bmp" -o -iname "*.tiff" \) | wc -l)

# if [ "$IMAGE_COUNT" -eq 0 ]; then
#     echo "❌ Error: No images found in $RESULTS_DIR"
#     exit 1
# fi

# echo "📊 Found $IMAGE_COUNT images to process in $RESULTS_DIR"
# echo ""
# echo "🖱️  GUI Instructions:"
# echo "   • For each image, a window will open for bounding box selection"
# echo "   • Click and drag to draw bounding boxes around animals"
# echo "   • Press 'r' to reset all bounding boxes"
# echo "   • Press 'u' to undo last bounding box"
# echo "   • Press 'Enter' to confirm and proceed to next image"
# echo "   • Press 'Esc' to skip current image"
# echo ""
# echo "🚀 Starting interactive processing..."

# # Run ViTPose inference with GUI mode and save bounding boxes
# CUDA_VISIBLE_DEVICES= python infer.py \
#     --cfg configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_large_ap10k_256x192.py \
#     --checkpoint checkpoints/ap10k.pth \
#     --img-dir "$RESULTS_DIR" \
#     --out-dir ./pose_results_interactive \
#     --gui-mode \
#     --save-bbox \
#     --bbox-dir ./bboxes_interactive \
#     --confidence 0.5 \
#     --draw-bbox \
#     --bbox-padding 20 \
#     --bbox-color "0,255,0" \
#     --bbox-thickness 3 \
#     --debug

# popd > /dev/null

# # Completion message with summary
# echo ""
# echo "✅ Pipeline completed successfully!"
# echo "📁 Results saved in:"
# echo "   • Mask predictions: /mnt/zone/B/Sayak/predictions_image/"
# echo "   • Inpainted images: /mnt/zone/B/Sayak/crfill/results/"
# echo "   • Pose estimations: /mnt/zone/B/Sayak/ViTPose/pose_results_interactive/"
# echo "   • Bounding boxes: /mnt/zone/B/Sayak/ViTPose/bboxes_interactive/"
# echo "   • Keypoint data: /mnt/zone/B/Sayak/ViTPose/pose_results_interactive/*_keypoints.json"
