Paper: https://arxiv.org/abs/2512.07712v2

# UnCageNet

UnCageNet is an animal cage removal pipeline that combines mask prediction, inpainting, and pose estimation to process images and videos of caged animals.

## Models
The models are in  https://huggingface.co/sayak-iit/UnCageNet

## Setting up `crfill/` and `ViTPose/`
1. **CR-Fill (Inpainting)**
    - Repository: [CR-Fill (Official)](https://github.com/lyndonzheng/CR-Fill)
    - To clone into the `crfill/` directory:
      ```bash
      git clone https://github.com/zengxianyu/crfill
      ```
    - Follow the installation instructions in the CR-Fill repository to install any additional dependencies.
2. **ViTPose (Pose Estimation)**
    - Repository: [ViTPose (Official)](https://github.com/ViTAE-Transformer/ViTPose)
    - To clone into the `ViTPose/` directory:
      ```bash
      git clone https://github.com/ViTAE-Transformer/ViTPose.git ViTPose
      ```
    - Follow the installation instructions in the ViTPose repository to install any additional dependencies.

## Pipeline Overview

The complete pipeline consists of three main steps:

1. **Mask Prediction** - Using `predict.py` to detect and create masks for cage structures
2. **Inpainting** - Using CR-Fill to remove cages and inpaint the masked regions
3. **Pose Estimation** - Using ViTPose via `infer_frame.py` to estimate animal poses

## Prerequisites

- Conda environments with required dependencies:
  - `deepfill` environment for mask prediction and inpainting
  - `myenvi` environment for pose estimation (or create from `myenvi.yml`)
- CUDA-enabled GPU (recommended)
- Pre-trained models:
  - Model for mask prediction (e.g., `model.pth`)
  - CR-Fill inpainting model (place in `crfill/` directory)
  - ViTPose checkpoint for animal pose estimation (e.g., `ap10k.pth`)

### Environment Setup

Create the required conda environment:

```bash
conda env create -f myenvi.yml
```

**Note:** The `crfill/` and `ViTPose/` directories should contain the respective inpainting and pose estimation code. These are typically submodules or separate repositories that need to be set up separately.

## Running the Pipeline

### Option 1: Using Shell Scripts (Recommended)

For automated processing, use the provided shell scripts:

#### Processing Images
```bash
./image.sh
```

#### Processing Videos
```bash
./video.sh
```

### Option 2: Manual Step-by-Step Execution

If you prefer to run each step manually, follow these instructions:

#### Step 1: Mask Prediction with predict.py

Create masks for cage structures in your images:

```bash
conda activate deepfill

python predict.py \
    --model_path /path/to/model.pth \
    --images_dir /path/to/input/images \
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
```

**Key Parameters:**
- `--model_path`: Path to your trained mask prediction model
- `--images_dir`: Directory containing input images
- `--output_dir`: Directory where masks will be saved
- `--backbone`: CNN backbone architecture (e.g., resnet101)
- `--confidence_boost`: Boost confidence values for better mask detection
- `--threshold`: Threshold for mask binarization
- `--apply_dilation`: Apply morphological dilation to masks
- `--dilate_kernel_size`: Size of dilation kernel
- `--dilate_iterations`: Number of dilation iterations

**Output:** Enhanced masks will be saved in `predictions_image/enhanced_masks/`

#### Step 2: Inpainting with CR-Fill

Remove cages and inpaint the masked regions:

```bash
conda activate deepfill

cd crfill

python test.py \
    --batchSize 1 \
    --nThreads 1 \
    --name objrmv \
    --dataset_mode testimage \
    --image_dir /path/to/input/images \
    --mask_dir /path/to/predictions_image/enhanced_masks \
    --output_dir ./results \
    --model inpaint \
    --netG baseconv \
    --which_epoch latest \
    --load_baseg

cd ..
```

**Key Parameters:**
- `--image_dir`: Directory containing original input images
- `--mask_dir`: Directory containing masks from Step 1
- `--output_dir`: Directory where inpainted images will be saved
- `--model`: Model type (inpaint)
- `--netG`: Generator network architecture

**Output:** Inpainted images will be saved in `crfill/results/`

#### Step 3: Pose Estimation with infer_frame.py

Estimate animal poses on the inpainted images:

```bash
conda activate myenvi

python infer_frame.py \
    --cfg configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_large_ap10k_256x192.py \
    --checkpoint checkpoints/ap10k.pth \
    --img-dir /path/to/crfill/results \
    --out-dir ./pose_results \
    --confidence 0.3 \
    --debug
```

**Key Parameters:**
- `--cfg`: Path to ViTPose configuration file
- `--checkpoint`: Path to ViTPose model checkpoint
- `--img-dir`: Directory containing inpainted images from Step 2
- `--out-dir`: Directory where pose estimation results will be saved
- `--confidence`: Confidence threshold for pose detection
- `--debug`: Enable debug mode for verbose output

**Output:** Pose estimation results with keypoints will be saved in the specified output directory

## Complete Pipeline Example

Here's a complete example workflow:

```bash
# Step 1: Create masks
conda activate deepfill
CUDA_VISIBLE_DEVICES=0 python predict.py \
    --model_path model.pth \
    --images_dir ./input_images \
    --output_dir predictions_image \
    --backbone resnet101 \
    --decoder_channels "1024,512,256,128,64" \
    --threshold 0.2 \
    --apply_dilation \
    --dilate_kernel_size 2 \
    --dilate_iterations 1

# Step 2: Inpaint images
cd crfill
CUDA_VISIBLE_DEVICES=0 python test.py \
    --batchSize 1 \
    --dataset_mode testimage \
    --image_dir ../input_images \
    --mask_dir ../predictions_image/enhanced_masks \
    --output_dir ./results \
    --model inpaint \
    --netG baseconv \
    --which_epoch latest \
    --load_baseg
cd ..

# Step 3: Estimate poses
conda activate myenvi
CUDA_VISIBLE_DEVICES=0 python infer_frame.py \
    --cfg configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_large_ap10k_256x192.py \
    --checkpoint checkpoints/ap10k.pth \
    --img-dir ./crfill/results \
    --out-dir ./pose_results \
    --confidence 0.3
```

## Directory Structure

```
UnCageNet/
├── predict.py              # Mask prediction script
├── infer_frame.py          # Pose estimation script
├── image.sh                # Automated image processing script
├── video.sh                # Automated video processing script
├── crfill/                 # CR-Fill inpainting module
│   └── test.py            # Inpainting script
├── ViTPose/               # ViTPose pose estimation module
├── predictions_image/     # Output directory for masks
└── README.md              # This file
```

## Notes

- Ensure all paths in the scripts are adjusted to match your local setup
- GPU memory requirements vary based on image resolution and batch size
- The pipeline can be customized by adjusting parameters in each step
- For video processing, refer to `video.sh` for batch processing examples

## Troubleshooting

- **Out of Memory Errors**: Reduce batch size or image resolution
- **Missing Models**: Ensure all model checkpoints are downloaded and paths are correct
- **Conda Environment Issues**: Verify that all required environments are created and activated correctly

```bibtex
@article{dutta2025uncagenet,
  title   = {UnCageNet: Tracking and Pose Estimation of Caged Animal},
  author  = {Dutta, Sayak and Katti, Harish and Verma, Shashikant and Raman, Shanmuganathan},
  journal = {arXiv preprint arXiv:2512.07712},
  year    = {2025},
  url     = {https://arxiv.org/abs/2512.07712}
}

