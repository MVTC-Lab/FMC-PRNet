# FMC-PRNet: Frequency-Modulated Coupled Progressive Refinement Network for Real-Time Aerial Small Object Detection

This code implements FMC-PRNet, an advanced and lightweight object detection framework specifically tailored for complex aerial and high-altitude imagery. The model fundamentally addresses the severe information loss of tiny objects in deep networks by integrating Enhanced Slice Sampling (ESSamp), Frequency-Modulated Coupling (FMC), Residual Atrous Spatial Pyramid Pooling (ResASPP), and a Progressive Refinement Network (PRNet). These features can typically be used in the following potential application scenarios:
- Real-time UAV (Unmanned Aerial Vehicle) edge-computing deployment.
- Power grid and equipment inspection (e.g., small insulator defect detection).
- High-altitude aerial object detection under complex backgrounds (urban traffic, maritime ships, dense crowds).
- Scenarios requiring "Train from Scratch" capability without dependency on massive pre-trained weights.

This approach was used to achieve state-of-the-art real-time small object detection performance on VisDrone2019 (approaching 68% mAP@0.5 for the Large variant), AI-TOD, and SIMD, achieving an optimal Pareto balance between accuracy and inference speed (up to 93.6 FPS for the Nano variant), as described in:

[paper link](Paper Link (Coming Soon)).

<div align="center">
  <img src="fig2.png" alt="FMC-PRNet Architecture" width="90%">
</div>


## Requirement
Python >= 3.8
Pytorch >= 1.9.0
torchvision >= 0.10.0        

## Training       
1. Create model_save_dir 
```                           
mkdir model_save_dir
```

2. Preprocessing   
```
$ mkdir data
$ cd data
```
## 📂 Data Preparation
You can download the original datasets from their official repositories:
- **[VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset)**: A large-scale drone-based dataset.
- **[AI-TOD](https://github.com/jwwangchn/AI-TOD)**: Aerial Imagery for Tiny Object Detection.
- **[SIMD](<Link_to_SIMD_dataset>)**: Satellite Imagery Multi-vehicles Dataset. *(Note: Replace with your specific download link if you host a pre-processed version)*

After downloading, we strictly reorganized the storage structure of images and labels into the standard YOLO format to avoid `AssertionError` during data loading. Please structure your directories exactly as follows:

```text
dataset_dir/
├── images/
│   ├── train/  <-- (Contains .jpg files)
│   └── val/
└── labels/
    ├── train/  <-- (Contains .txt files)
    └── val/
```
Please modify the train and val paths in your specific dataset YAML file (e.g., SIMD.yaml, AI-TOD.yaml) to point to the images/train and images/val directories. The data loader will automatically locate the corresponding labels directory.

Multi-GPU Training (Train from Scratch)

FMC-PRNet demonstrates exceptional endogenous learning capabilities and is trained from scratch without external pre-trained weights. To launch distributed training across multiple GPUs, use the torchrun utility.

Example for training the Small (S) variant on VisDrone using 2 GPUs:

```
CUDA_VISIBLE_DEVICES=0,1 PYTHONWARNINGS="ignore" NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nproc_per_node 2 --master_port 29500 tools/train.py --batch-size 16 --conf-file ./configs/fmc_prnet-s.py --data-path ./data/VisDrone.yaml --epochs 400 --img-size 1280 --fuse_ab --use_syncbn --device 0,1 --name FMC_PRNet_S_VisDrone --output-dir ./result --workers 4
```

Single-GPU Training：

```
python tools/train.py --batch-size 16 --conf-file ./configs/fmc_prnet-n.py --data-path ./data/SIMD.yaml --epochs 400 --img-size 800 --device 0 --name FMC_PRNet_N_SIMD --output-dir ./result
```

Evaluation

To evaluate a trained model on the validation or test set:

```
python tools/eval.py --conf-file ./configs/fmc_prnet-l.py --data-path ./data/AI-TOD.yaml --weights ./result/FMC_PRNet_L_Final/weights/best.pt --img-size 800 --device 0
```

Citation information will be updated soon.
