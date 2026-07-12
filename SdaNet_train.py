#
#
#
## -*- coding: utf-8 -*-
#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.utils.data import Dataset, DataLoader
#import torch.nn.functional as F
#import numpy as np
#import cv2
#import os
#import glob
#import time
#import sys
#import traceback
#import shutil
#from torchvision import transforms
#import argparse
#from concurrent.futures import ThreadPoolExecutor
#import torch.optim.lr_scheduler as lr_scheduler
#
## Import SdaNet model
#from model.SdaNet import SdaNet
#
#num_gpus = torch.cuda.device_count()
#use_data_parallel = num_gpus >= 2
#print(f"Detected {num_gpus} GPUs (Use DataParallel: {use_data_parallel})")
#
#
#def get_parser():
#    parser = argparse.ArgumentParser(description='SdaNet Training Script (Stroke Analysis)')
#
#    # 1. Data parameters
#    parser.add_argument('--data_dir', type=str,
#                        default="/media/text/afedbba0-93b4-490d-9d12-20e5b7b22aa7/yan/exp2/S-Dauble-Attention/dataset/strock_semantics/semantics_kai/semantics_train",
#                        help='Training data directory')
#    parser.add_argument('--val_data_dir', type=str,
#                        default="/media/text/afedbba0-93b4-490d-9d12-20e5b7b22aa7/yan/exp2/S-Dauble-Attention/dataset/strock_semantics/semantics_kai/semantics_val",
#                        help='Validation data directory (use test set as validation set)')
##    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
#    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
#    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading threads')
#
#    # 2. Training parameters
#    parser.add_argument('--epochs', type=int, default=200, help='Total training epochs')
#  
#    parser.add_argument('--initial_lr', type=float, default=0.0005, help='Initial learning rate')
#    parser.add_argument('--num_stroke_classes', type=int, default=8, help='Number of stroke classes')
#
#    # Loss function parameters
#  
#    parser.add_argument('--class_weights', type=str,
#                        default="1.2,1.1,1.1,1.0,1.0,1.1,1.1,1.2",
#                        help='Class weights (comma-separated, must match number of classes)')
#
#    #  Model parameters
#    parser.add_argument('--model_save_dir', type=str,
#                        default="/media/text/afedbba0-93b4-490d-9d12-20e5b7b22aa7/yan/exp2/S-Dauble-Attention/dataset/weight/kai",
#                        help='Model save directory')
#    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
#    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
#
#    # 4. GPU settings
#    parser.add_argument('--gpu_ids', type=str, default="0,1", help='GPU IDs to use, e.g., "0,1"')
#
#    # 5. Resume training parameters
#    parser.add_argument('--continue_training', action='store_true', default=False,
#                        help='Resume training from checkpoint')
#    parser.add_argument('--pretrained_model_path', type=str, default="", help='Pretrained model path')
#    parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch')
#    parser.add_argument('--num_epochs_continue', type=int, default=100, help='Number of epochs to resume')
#
#    # 6. Loss function type (fixed to weighted BCE)
#    parser.add_argument('--loss_type', type=str, default='weighted_bce', choices=['weighted_bce'],
#                        help='Loss function type')
#
#    # 7. Visualization parameters
#    parser.add_argument('--visualize_frequency', type=int, default=99, help='Visualization save frequency (epochs)')
#    parser.add_argument('--target_size', type=int, default=1024, help='Visualization image target size')
#    parser.add_argument('--border_width', type=int, default=64, help='Visualization border width')
#    parser.add_argument('--vis_filtered_samples', action='store_true', default=True,
#                        help='Visualize samples filtered due to empty non-cross regions')
#                        
#                        
#                        
##        only ssra              
##    parser.add_argument('--use_ssca', type=lambda x: x.lower() == 'true', default=True,
##                    help='Use SSCA (default: True)')
#
#
##          no attention
##    parser.add_argument('--use_attention', type=lambda x: x.lower() == 'true', default=True,
##                        help='Use double attention (default: True)')
#                        
#    return parser
#
#
#class FastStrokeDataset(Dataset):
#    """Fast loading stroke dataset (including cross region, non-cross region, original image)"""
#
#    def __init__(self, data_dir, transform=None, num_workers=8, preload=False):
#        self.data_dir = data_dir
#        self.transform = transform
#        self.num_workers = num_workers
#        self.preload = preload
#
#        if not os.path.exists(data_dir):
#            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
#
#        # Find all image folders
#        self.image_folders = [f for f in glob.glob(os.path.join(data_dir, "image_*")) if os.path.isdir(f)]
#        self.image_folders.sort()  # Ensure consistent order
#        print(f"Found {len(self.image_folders)} image folders in {data_dir}")
#
#        # Build dataset file list (add non-cross region image loading)
#        self.data = []
#        for i, folder in enumerate(self.image_folders):
#            folder_name = os.path.basename(folder)  # e.g., "image_001"
#            try:
#                # Find original image, cross region image, non-cross region image and label
#                original_files = glob.glob(os.path.join(folder, "*union*"))  # Original character image
#                cross_files = glob.glob(os.path.join(folder, "*intersection*"))  # Cross region image
#                non_cross_files = glob.glob(os.path.join(folder, "*processed*"))  # Non-cross region image (added)
#                label_files = glob.glob(os.path.join(folder, "*label.npy"))  # Label
#
#                if original_files and cross_files and non_cross_files and label_files:
#                    self.data.append({
#                        "original": original_files[0],
#                        "cross": cross_files[0],  # Cross region image
#                        "non_cross": non_cross_files[0],  # Non-cross region image (added)
#                        "label": label_files[0],
#                        "folder_name": folder_name
#                    })
#                    if i < 5 or i % 1000 == 0:
#                        print(f"Loading data, {i}/{len(self.image_folders)}: {folder_name}")
#                else:
#                    missing = []
#                    if not original_files:
#                        missing.append("union.png (original image)")
#                    if not cross_files:
#                        missing.append("_intersection.png (cross region image)")
#                    if not non_cross_files:
#                        missing.append("_final_processed.png (non-cross region image)")  # Added check
#                    if not label_files:
#                        missing.append("label.npy (label)")
#                    print(f"Warning: Folder {folder_name} missing files: {missing}, skipped")
#            except Exception as e:
#                print(f"Error processing folder {folder_name}: {e}")
#
#        print(f"Total samples in {data_dir} dataset: {len(self.data)}")
#
#        # Preload data (if needed)
#        if preload:
#            self._preload_data()
#
#    def _preload_data(self):
#        """Preload all data into memory"""
#        print(f"Preloading data with {self.num_workers} threads...")
#        start_time = time.time()
#        try:
#            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
#                self.preloaded_data = list(executor.map(lambda x: self.data[x], range(len(self.data))))
#        except Exception as e:
#            print(f"Data preloading error: {e}")
#            traceback.print_exc()
#            raise
#        end_time = time.time()
#        print(f"Data preloading completed, {len(self.preloaded_data)} samples, time taken: {end_time - start_time:.2f}s")
#
#    def __len__(self):
#        return len(self.data) if not self.preload else len(self.preloaded_data)
#
#    def __getitem__(self, idx):
#        if self.preload:
#            return self.preloaded_data[idx]
#        return self.data[idx]
#
#
#class SdaNetTrainer:
#    """SdaNet Trainer (supports filtered sample visualization)"""
#
#    def __init__(self, args):
#        self.args = args
#        self.best_val_iou = -float('inf')  # Best IoU (initialized to negative infinity)
#        self.best_model_path = os.path.join(args.model_save_dir, "best_model.pth")  # Best model path
#        self.best_vis_dir = os.path.join(args.model_save_dir, "best_visualization")  # Best visualization directory
#        self.best_info_path = os.path.join(args.model_save_dir, "best_info.txt")  # Best model info
#        self.start_time = time.time()
#        
#        
#        # Filtered sample visualization directory (with timestamp)
#        self.filtered_vis_dir = os.path.join(args.model_save_dir,
#                                             f"filtered_samples_vis_{time.strftime('%Y%m%d_%H%M%S')}")
#        if args.vis_filtered_samples:
#            os.makedirs(self.filtered_vis_dir, exist_ok=True)
#            print(f"Filtered samples (empty non-cross regions) will be saved to: {self.filtered_vis_dir}")
#
#        # Parse class weights (convert to tensor and move to device)
#        self.class_weights = torch.tensor(
#            [float(w) for w in args.class_weights.split(',')],
#            dtype=torch.float32
#        )
#        # Verify class weight count matches number of classes
#        assert len(self.class_weights) == args.num_stroke_classes, \
#            f"Number of class weights must match number of classes ({args.num_stroke_classes}), got {len(self.class_weights)}"
#        print(f"Class weights: {self.class_weights.numpy()}")
#
#        # Visualization parameters
#        self.target_size = (args.target_size, args.target_size)  # Target size
#        self.border_color = (230, 230, 10)  # Border color (yellow)
#        self.border_width = args.border_width  # Border width
#        # Generate fixed colors (seed 42 ensures consistency)
#        np.random.seed(42)
#        self.colors = np.random.randint(50, 256, size=(args.num_stroke_classes, 3), dtype=np.uint8)
#        print(f"Generated {args.num_stroke_classes} colors, target size: {self.target_size[0]}x{self.target_size[1]}")
#
#        print("Trainer initialization completed")
#
#        # Set device
#        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#        self.class_weights = self.class_weights.to(self.device)  # Move class weights to device
#        print(f"Using device: {self.device}")
#
#        # Check GPU count
#        self.num_gpus = torch.cuda.device_count()
#        if self.num_gpus > 1:
#            print(f"Detected {self.num_gpus} GPUs, using DataParallel for multi-GPU training")
#        else:
#            print(f"Detected {self.num_gpus} GPU, using single GPU training")
#
#        # Create save directory
#        os.makedirs(args.model_save_dir, exist_ok=True)
#        print(f"Model save directory: {args.model_save_dir}")
#
#        # Initialize model
#        print(f"Initializing SdaNet")
#        self.model = SdaNet(
#            embed_dim=args.embed_dim,
#            num_heads=args.num_heads,
#            num_stroke_classes=args.num_stroke_classes,
#            
##        only ssra
##            use_ssca=args.use_ssca 
#
#
##          no attention 
##        use_attention=args.use_attention
#            
#        ).to(self.device)
#        print("Model initialization completed")
#
#        # Check memory status
#        try:
#            if torch.cuda.is_available():
#                torch.cuda.empty_cache()
#                for i in range(self.num_gpus):
#                    mem_info = torch.cuda.mem_get_info(i)
#                    print(
#                        f"GPU {i} memory: Free {mem_info[0] / 1024 ** 3:.2f} GB, Total {mem_info[1] / 1024 ** 3:.2f} GB")
#        except Exception as e:
#            print(f"Memory check error: {e}")
#            raise
#
#        # Set up multi-GPU
#        if self.num_gpus > 1:
#            try:
#                self.model = nn.DataParallel(self.model)
#                print(f"DataParallel setup successful, using {self.num_gpus} GPUs")
#            except Exception as e:
#                print(f"DataParallel setup failed: {e}")
#                raise
#
#        # Set up Adam optimizer
#       
#        self.optimizer = optim.Adam(
#            self.model.parameters(),
#            lr=args.initial_lr,
#            betas=(0.9, 0.999),  
#            weight_decay=1e-5    
#        )
#        print("Optimizer setup completed (Adam)")
#
#        # Set up learning rate scheduler
#      
#        self.scheduler = lr_scheduler.StepLR(
#            self.optimizer,
#            step_size=30,  
#            gamma=0.8      
#        )
#        print(f"Learning rate scheduler setup completed, time taken: {time.time() - self.start_time:.2f}s")
#
#    def _get_real_model(self):
#        """Get real model (handle DataParallel wrapping)"""
#        if isinstance(self.model, torch.nn.DataParallel):
#            return self.model.module
#        return self.model
#
#    def _save_filtered_visualization(self, folder_name, original_img, cross_img, non_cross_img):
#        """Save visualization of filtered samples (added non-cross region image)"""
#        if not self.args.vis_filtered_samples:
#            return
#
#        try:
#            # Ensure directory exists
#            os.makedirs(self.filtered_vis_dir, exist_ok=True)
#
#            # Original image (128x128)
#            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
#            cv2.imwrite(os.path.join(self.filtered_vis_dir, f"{folder_name}_original.png"), original_rgb)
#
#            # Cross region image
#            cross_rgb = cv2.cvtColor(cross_img, cv2.COLOR_GRAY2BGR)
#            cv2.imwrite(os.path.join(self.filtered_vis_dir, f"{folder_name}_cross_region.png"), cross_rgb)
#
#            # Non-cross region image (added)
#            non_cross_rgb = cv2.cvtColor(non_cross_img, cv2.COLOR_GRAY2BGR)
#            cv2.imwrite(os.path.join(self.filtered_vis_dir, f"{folder_name}_non_cross_region.png"), non_cross_rgb)
#
#            # Empty non-cross region marker image (red border + text)
#            empty_vis = np.ones_like(non_cross_img) * 255
#            empty_vis = cv2.cvtColor(empty_vis, cv2.COLOR_GRAY2BGR)
#            cv2.rectangle(empty_vis, (0, 0), (empty_vis.shape[1] - 1, empty_vis.shape[0] - 1), (0, 0, 255), 2)  # Red border
#            cv2.putText(empty_vis, "Empty non-cross", (10, 30),
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
#            cv2.imwrite(os.path.join(self.filtered_vis_dir, f"{folder_name}_non_cross_empty.png"), empty_vis)
#
#            print(f"Filtered sample visualization saved: {folder_name}")
#
#        except Exception as e:
#            print(f"Failed to save visualization for filtered sample {folder_name}: {e}")
#
#    # -------------------------- Core loss function implementation --------------------------
#    def weighted_bce_loss(self, pred_logits, target):
#        """Weighted BCE loss with class weights"""
#        # Calculate raw BCE loss (unnormalized)
#        bce = F.binary_cross_entropy_with_logits(
#            pred_logits,
#            target,
#            reduction='none'  # Keep loss for each element
#        )  # Shape [B, C, H, W]
#
#        # Expand class weight dimensions to match loss shape [B, C, H, W]
#        weights = self.class_weights.view(1, -1, 1, 1)  # [1, C, 1, 1]
#        weighted_bce = bce * weights  # Apply class weights
#
#        # Calculate weighted average loss
#        return weighted_bce.mean()
#
#    def combined_loss(self, pred_logits, target):
#        """Only use weighted BCE loss"""
#        bce_loss = self.weighted_bce_loss(pred_logits, target)
#        return bce_loss, bce_loss, 0.0  # Keep return format compatible, last value is 0
#
#    # ----------------------------------------------------------------------
#
#    def process_sample(self, sample):
#        """Process single sample, prepare three inputs and label (added non-cross region processing)"""
#        # Load original image, cross region image, non-cross region image and label
#        original_img = cv2.imread(sample["original"], cv2.IMREAD_GRAYSCALE)
#        cross_img = cv2.imread(sample["cross"], cv2.IMREAD_GRAYSCALE)  # Cross region image
#        non_cross_img = cv2.imread(sample["non_cross"], cv2.IMREAD_GRAYSCALE)  # Non-cross region image (added)
#        label = np.load(sample["label"])  # Label shape: (H, W, num_stroke_classes)
#        folder_name = sample["folder_name"]  # Folder name
#        # Load original image, cross region image, non-cross region image (ensure grayscale)
#
#
#        # Check image loading status
#        if original_img is None:
#            raise ValueError(f"Failed to load original image, folder: {folder_name}, path: {sample['original']}")
#        if cross_img is None:
#            raise ValueError(f"Failed to load cross region image, folder: {folder_name}, path: {sample['cross']}")
#        if non_cross_img is None:  # Added check
#            raise ValueError(f"Failed to load non-cross region image, folder: {folder_name}, path: {sample['non_cross']}")
#
#        if original_img.shape != (128, 128):
#            original_img = cv2.resize(original_img, (128, 128), interpolation=cv2.INTER_NEAREST)
#
#
#        if cross_img.shape != (128, 128):
#            cross_img = cv2.resize(cross_img, (128, 128), interpolation=cv2.INTER_NEAREST)
#
#
#        if non_cross_img.shape != (128, 128):
#            non_cross_img = cv2.resize(non_cross_img, (128, 128), interpolation=cv2.INTER_NEAREST)
#
#
#        # Resize label to 128x128
#        if label.shape[0] != 128 or label.shape[1] != 128:
#            label_resized = np.zeros((128, 128, label.shape[2]), dtype=label.dtype)
#            for c in range(label.shape[2]):
#                label_resized[:, :, c] = cv2.resize(label[:, :, c], (128, 128), interpolation=cv2.INTER_NEAREST)
#            label = label_resized
#
#        # Prepare input tensors (normalized to [0,1], add channel dimension)
#        original_tensor = torch.from_numpy(original_img).float().unsqueeze(0) / 255.0  # [1, 128, 128]
#        cross_tensor = torch.from_numpy(cross_img).float().unsqueeze(0) / 255.0  # [1, 128, 128]
#        non_cross_tensor = torch.from_numpy(non_cross_img).float().unsqueeze(0) / 255.0  # [1, 128, 128] (added)
#
#        # Prepare label: [num_classes, 128, 128] (from [H,W,C] to [C,H,W])
#        label_tensor = torch.from_numpy(label).float().permute(2, 0, 1)  # [num_classes, 128, 128]
#
#        return original_tensor, cross_tensor, non_cross_tensor, label_tensor, folder_name
#
#    # -------------------------- Visualization functions --------------------------
#
#    def get_color_image(self, single_image, white_bg=True, apply_threshold=True):
#        """Convert segmentation result to color image"""
#        if isinstance(single_image, torch.Tensor):
#            single_image = single_image.detach().cpu().numpy()
#
#        # Process logits (apply sigmoid threshold)
#        if apply_threshold and len(single_image.shape) == 3:  # [num_classes, H, W]
#            single_image = (torch.sigmoid(torch.from_numpy(single_image)) > 0.5).numpy()
#        elif apply_threshold and len(single_image.shape) == 3:  # Already probabilities
#            single_image = (single_image > 0.5).numpy()
#
#        # Ensure input is one-hot encoded [num_classes, H, W]
#        if len(single_image.shape) != 3:
#            raise ValueError(f"Color mapping requires 3D tensor, got {single_image.shape}")
#
#        C, H, W = single_image.shape
#
#        # Create background
#        color_img = np.full((H, W, 3), 255, dtype=np.uint8) if white_bg else np.zeros((H, W, 3), dtype=np.uint8)
#
#        # Add color for each class (multi-label support)
#        for c in range(C):
#            if c >= len(self.colors):
#                continue  # Prevent index out of range for color list
#
#            mask = single_image[c] > 0.5  # Binary mask
#            color = self.colors[c]  # Fixed color
#
#            # Apply color to RGB image (prevent overflow 255)
#            current_color = color_img[mask]
#            new_color = current_color + color
#            new_color = np.clip(new_color, 0, 255)
#            color_img[mask] = new_color
#
#        return color_img
#
#    def add_border(self, img, color, width):
#        """Add border to image"""
#        h, w = img.shape[:2]
#        return cv2.rectangle(img.copy(), (0, 0), (w - 1, h - 1), color, thickness=width)
#
#    # -------------------------- Visualization functions --------------------------
#
#    def visualize_epoch_predictions(self, epoch):
#        """Visualize validation set predictions for current epoch (adapted for three inputs)"""
#        # Create visualization directory for current epoch
#        epoch_vis_dir = os.path.join(self.args.model_save_dir, f"epoch_{epoch}_visualization")
#        if os.path.exists(epoch_vis_dir):
#            shutil.rmtree(epoch_vis_dir)  # Delete existing directory
#        os.makedirs(epoch_vis_dir, exist_ok=True)
#
#        test_dataset = self.val_loader.dataset
#        total_samples = len(test_dataset)
#        print(
#            f"\n[Visualization] Starting epoch {epoch} visualization, processing {total_samples} samples, save path: {epoch_vis_dir}")
#
#        self.model.eval()
#        with torch.no_grad():
#            # Process sample by sample
#            for vis_idx, data_idx in enumerate(range(total_samples)):
#                try:
#                    # Load single sample
#                    sample = test_dataset[data_idx]
#                    result = self.process_sample(sample)
#
#                    if result is None:
#                        print(
#                            f"[Visualization Error] Sample {sample.get('folder_name', 'unknown')} filtered (empty non-cross region)")
#                        continue
#
#                    original_img, cross_img, non_cross_img, label, folder_name = result  # Added non_cross_img
#
#                    # Model inference (pass three inputs)
#                    original_batch = original_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    cross_batch = cross_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    non_cross_batch = non_cross_img.unsqueeze(0).to(self.device, non_blocking=True)  # Added
#                    pred_logits, valid_mask = self.model(cross_batch, non_cross_batch, original_batch)  # Correct input order
#
#                    # Process multi-label predictions (apply sigmoid)
#                    pred_prob = torch.sigmoid(pred_logits)
#
#                    # Generate visualization images (target size)
#                    
#                    original_np = original_img.squeeze().cpu().numpy()  # (128,128), range [0,1]
#                    original_rgb = cv2.cvtColor((original_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    original_rgb_target = cv2.resize(original_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    # Cross region image visualization (added)
#                    cross_np = cross_img.squeeze().cpu().numpy()
#                    cross_rgb = cv2.cvtColor((cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    cross_rgb_target = cv2.resize(cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    # Non-cross region image visualization (added)
#                    non_cross_np = non_cross_img.squeeze().cpu().numpy()
#                    non_cross_rgb = cv2.cvtColor((non_cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    non_cross_rgb_target = cv2.resize(non_cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    # Label visualization (one-hot format)
#                    label_np = label.cpu().numpy()  # (num_classes, 128, 128)
#                    label_color = self.get_color_image(label_np, white_bg=True, apply_threshold=False)
#                    label_color_target = cv2.resize(label_color, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    # Prediction visualization
#                    pred_np = pred_prob.squeeze(0).cpu().numpy()  # (num_classes, 128, 128)
#                    pred_color = self.get_color_image(pred_np, white_bg=True, apply_threshold=True)
#                    pred_color_target = cv2.resize(pred_color, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    # Add borders
#                    original_with_border = self.add_border(original_rgb_target, self.border_color, self.border_width)
#                    cross_with_border = self.add_border(cross_rgb_target, self.border_color, self.border_width)  # Added
#                    non_cross_with_border = self.add_border(non_cross_rgb_target, self.border_color, self.border_width)  # Added
#                    label_with_border = self.add_border(label_color_target, self.border_color, self.border_width)
#                    prediction_with_border = self.add_border(pred_color_target, self.border_color, self.border_width)
#
#                    # Save paths
#                    save_paths = {
#                        "original": os.path.join(epoch_vis_dir,
#                                                 f"{folder_name}_original_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "cross": os.path.join(epoch_vis_dir,  # Added
#                                              f"{folder_name}_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "non_cross": os.path.join(epoch_vis_dir,  # Added
#                                                 f"{folder_name}_non_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "label": os.path.join(epoch_vis_dir,
#                                              f"{folder_name}_label_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "prediction": os.path.join(epoch_vis_dir,
#                                                   f"{folder_name}_prediction_{self.target_size[0]}x{self.target_size[1]}.png")
#                    }
#
#                    # Save images
#                    cv2.imwrite(save_paths["original"], original_with_border)
#                    cv2.imwrite(save_paths["cross"], cross_with_border)  # Added
#                    cv2.imwrite(save_paths["non_cross"], non_cross_with_border)  # Added
#                    cv2.imwrite(save_paths["label"], label_with_border)
#                    cv2.imwrite(save_paths["prediction"], prediction_with_border)
#
#                    # Verify save status
#                    for name, path in save_paths.items():
#                        saved_img = cv2.imread(path)
#                        if saved_img is None:
#                            print(
#                                f"[Visualization Error] Epoch {epoch} sample {folder_name} {name} save failed, path: {path}")
#                        elif saved_img.shape[:2] != (self.target_size[1], self.target_size[0]):
#                            print(
#                                f"[Visualization Error] Epoch {epoch} sample {folder_name} {name} size error, actual: {saved_img.shape[1]}x{saved_img.shape[0]}")
#
#                    # Print progress
#                    if (vis_idx + 1) % 10 == 0 or vis_idx == total_samples - 1:
#                        print(
#                            f"[Visualization] Epoch {epoch} progress: {vis_idx + 1}/{total_samples} samples, current folder: {folder_name}")
#
#                except Exception as e:
#                    print(f"[Visualization Error] Epoch {epoch} sample {folder_name} visualization failed: {str(e)}")
#                    traceback.print_exc()
#                    continue
#
#        print(f"[Visualization] Epoch {epoch} visualization completed, saved to: {epoch_vis_dir}\n")
#
#    def save_visualizations(self):
#        """Save prediction visualizations for best model (adapted for three inputs)"""
#        if os.path.exists(self.best_vis_dir):
#            shutil.rmtree(self.best_vis_dir)
#        os.makedirs(self.best_vis_dir, exist_ok=True)
#
#        # Load best model weights (ensure using best parameters for visualization)
#        try:
#            model_to_load = self._get_real_model()
#            model_to_load.load_state_dict(torch.load(self.best_model_path))
#            print(f"Loaded best model weights for visualization: {self.best_model_path}")
#        except Exception as e:
#            print(f"Failed to load best model weights, using current model state for visualization: {e}")
#
#        test_dataset = self.val_loader.dataset
#        total_samples = len(test_dataset)
#        print(
#            f"\n[Best Model Visualization] Generating visualizations for {total_samples} validation samples, save path: {self.best_vis_dir}")
#
#        self.model.eval()
#        with torch.no_grad():
#            # Process sample by sample
#            for vis_idx, data_idx in enumerate(range(total_samples)):
#                try:
#                    sample = test_dataset[data_idx]
#                    result = self.process_sample(sample)
#
#                    if result is None:
#                        print(
#                            f"[Best Model Visualization Error] Sample {sample.get('folder_name', 'unknown')} filtered (empty non-cross region)")
#                        continue
#
#                    original_img, cross_img, non_cross_img, label, folder_name = result  # Added non_cross_img
#
#                    # Model inference (pass three inputs)
#                    original_batch = original_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    cross_batch = cross_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    non_cross_batch = non_cross_img.unsqueeze(0).to(self.device, non_blocking=True)  # Added
#                    pred_logits, valid_mask = self.model(cross_batch, non_cross_batch, original_batch)  # Correct input order
#
#                    pred_prob = torch.sigmoid(pred_logits)  # Multi-label probabilities
#
#                    # Generate visualization images
#                    original_np = original_img.squeeze().cpu().numpy()
#                    original_rgb = cv2.cvtColor((original_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    original_rgb_target = cv2.resize(original_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    # Cross region image visualization (added)
#                    cross_np = cross_img.squeeze().cpu().numpy()
#                    cross_rgb = cv2.cvtColor((cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    cross_rgb_target = cv2.resize(cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    # Non-cross region image visualization (added)
#                    non_cross_np = non_cross_img.squeeze().cpu().numpy()
#                    non_cross_rgb = cv2.cvtColor((non_cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    non_cross_rgb_target = cv2.resize(non_cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    # Label visualization
#                    label_np = label.cpu().numpy()  # (num_classes, 128, 128)
#                    label_color = self.get_color_image(label_np, white_bg=True, apply_threshold=False)
#                    label_color_target = cv2.resize(label_color, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    # Prediction visualization
#                    pred_np = pred_prob.squeeze(0).cpu().numpy()
#                    pred_color = self.get_color_image(pred_np, white_bg=True, apply_threshold=False)
#                    pred_color_target = cv2.resize(pred_color, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    # Add borders
#                    original_with_border = self.add_border(original_rgb_target, self.border_color, self.border_width)
#                    cross_with_border = self.add_border(cross_rgb_target, self.border_color, self.border_width)  # Added
#                    non_cross_with_border = self.add_border(non_cross_rgb_target, self.border_color, self.border_width)  # Added
#                    label_with_border = self.add_border(label_color_target, self.border_color, self.border_width)
#                    prediction_with_border = self.add_border(pred_color_target, self.border_color, self.border_width)
#
#                    # Save paths
#                    save_paths = {
#                        "original": os.path.join(self.best_vis_dir,
#                                                 f"{folder_name}_original_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "cross": os.path.join(self.best_vis_dir,  # Added
#                                              f"{folder_name}_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "non_cross": os.path.join(self.best_vis_dir,  # Added
#                                                 f"{folder_name}_non_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "label": os.path.join(self.best_vis_dir,
#                                              f"{folder_name}_label_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "prediction": os.path.join(self.best_vis_dir,
#                                                   f"{folder_name}_prediction_{self.target_size[0]}x{self.target_size[1]}.png")
#                    }
#
#                    cv2.imwrite(save_paths["original"], original_with_border)
#                    cv2.imwrite(save_paths["cross"], cross_with_border)  # Added
#                    cv2.imwrite(save_paths["non_cross"], non_cross_with_border)  # Added
#                    cv2.imwrite(save_paths["label"], label_with_border)
#                    cv2.imwrite(save_paths["prediction"], prediction_with_border)
#
#                    # Print progress
#                    if (vis_idx + 1) % 10 == 0 or vis_idx == total_samples - 1:
#                        print(
#                            f"[Best Model Visualization] Progress: {vis_idx + 1}/{total_samples} samples, current folder: {folder_name}")
#
#                except Exception as e:
#                    print(f"[Best Model Visualization Error] Sample {folder_name} visualization failed: {str(e)}")
#                    traceback.print_exc()
#                    continue
#
#        print(f"[Best Model Visualization] All visualizations completed, saved to: {self.best_vis_dir}\n")
#
#    def calculate_iou(self, pred_logits, label, eps=1e-6):
#        """Calculate mean IoU for multi-label classification (with smoothing)"""
#        batch_size = pred_logits.size(0)
#        mean_ious = []
#
#        for b in range(batch_size):
#            pred_logits_b = pred_logits[b]  # [num_classes, H, W]
#            target_b = label[b]  # [num_classes, H, W]
#
#            # Calculate IoU per class
#            class_ious = []
#            for c in range(self.args.num_stroke_classes):
#                pred_binary = (torch.sigmoid(pred_logits_b[c]) > 0.5)  # Apply sigmoid threshold
#                target_binary = target_b[c]  # Already binary [0,1]
#
#                intersection = (pred_binary * target_binary).sum()
#                union = (pred_binary + target_binary).sum() - intersection
#
#                # Smoothing to avoid division by zero
#                if union == 0:
#                    iou = 1.0 if intersection == 0 else 0.0
#                else:
#                    iou = ((intersection + eps) / (union + eps)).item()
#                class_ious.append(iou)
#
#            # Only consider classes with foreground in target
#            valid_classes = [class_ious[c] for c in range(self.args.num_stroke_classes) if target_b[c].sum() > 0]
#
#            if valid_classes:
#                mean_ious.append(sum(valid_classes) / len(valid_classes))
#            else:
#                mean_ious.append(0.0)
#
#        return torch.tensor(mean_ious, device=pred_logits.device).mean(), mean_ious
#    
#    
#    
#    
#    def calculate_precision_recall_f1(self, pred_logits, label):
#        batch_size = pred_logits.size(0)
#        sample_precisions = []
#        sample_recalls = []
#        sample_f1s = []
#        eps = 1e-8
#
#        for b in range(batch_size):
#            pred_logits_b = pred_logits[b]
#            target_b = label[b]
#
#            pred_binary = (torch.sigmoid(pred_logits_b) > 0.5).float()
#            target_binary = target_b.float()
#
#            tp = (pred_binary * target_binary).sum().item()
#            fp = (pred_binary * (1 - target_binary)).sum().item()
#            fn = ((1 - pred_binary) * target_binary).sum().item()
#
#            precision = tp / (tp + fp + eps)
#            recall = tp / (tp + fn + eps)
#            f1 = 2 * precision * recall / (precision + recall + eps)
#
#            sample_precisions.append(precision)
#            sample_recalls.append(recall)
#            sample_f1s.append(f1)
#
#        batch_precision = sum(sample_precisions) / len(sample_precisions) if sample_precisions else 0.0
#        batch_recall = sum(sample_recalls) / len(sample_recalls) if sample_recalls else 0.0
#        batch_f1 = sum(sample_f1s) / len(sample_f1s) if sample_f1s else 0.0
#
#        return batch_precision, batch_recall, batch_f1  
#        
#        
##    def calculate_precision_recall_f1(self, pred_logits, label):
##        """Calculate precision, recall and F1 score for multi-label classification"""
##        batch_size = pred_logits.size(0)
##        sample_precisions = []
##        sample_recalls = []
##        sample_f1s = []
##
##        for b in range(batch_size):
##            pred_logits_b = pred_logits[b]  # [num_classes, H, W]
##            target_b = label[b]  # [num_classes, H, W]
##
##            # Apply sigmoid and threshold to get binary predictions
##            pred_binary = (torch.sigmoid(pred_logits_b) > 0.5)  # [num_classes, H, W]
##
##            # Calculate TP, FP, FN per class
##            class_tps = []
##            class_fps = []
##            class_fns = []
##
##            for c in range(self.args.num_stroke_classes):
##                # Only consider classes present in target
##                if target_b[c].sum() > 0:
##                    pred_c = pred_binary[c]  # [H, W]
##                    target_c = target_b[c]  # [H, W]
##
##                    tp = (pred_c * target_c).sum().item()
##                    fp = (pred_c * (1 - target_c)).sum().item()
##                    fn = ((1 - pred_c) * target_c).sum().item()
##
##                    class_tps.append(tp)
##                    class_fps.append(fp)
##                    class_fns.append(fn)
##
##            # Calculate sample-level metrics (class average)
##            total_tp = sum(class_tps)
##            total_fp = sum(class_fps)
##            total_fn = sum(class_fns)
##
##            # Calculate multi-label precision, recall, F1
##            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
##            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
##            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
##
##            sample_precisions.append(precision)
##            sample_recalls.append(recall)
##            sample_f1s.append(f1)
##
##        # Return batch-level average (sample-level average)
##        batch_precision = sum(sample_precisions) / len(sample_precisions) if sample_precisions else 0.0
##        batch_recall = sum(sample_recalls) / len(sample_recalls) if sample_recalls else 0.0
##        batch_f1 = sum(sample_f1s) / len(sample_f1s) if sample_f1s else 0.0
#
##        return batch_precision, batch_recall, batch_f1
#
#    def calculate_final_metrics(self):
#        """Calculate final stroke extraction metrics on validation set"""
#        self.model.eval()
#        all_precisions = []
#        all_recalls = []
#        all_f1s = []
#        total_samples = len(self.val_loader.dataset)
#
#        print(f"Evaluating final metrics on {total_samples} validation samples...")
#
#        with torch.no_grad():
#            # Process all samples in batches
#            for batch_idx, batch in enumerate(self.val_loader):
#                try:
#                    # Process batch samples
#                    original_list = []
#                    cross_list = []
#                    non_cross_list = []  # Added
#                    labels_list = []
#
#                    for i in range(len(batch["original"])):
#                        sample = {
#                            "original": batch["original"][i],
#                            "cross": batch["cross"][i],  # Cross region image
#                            "non_cross": batch["non_cross"][i],  # Non-cross region image (added)
#                            "label": batch["label"][i],
#                            "folder_name": batch["folder_name"][i]
#                        }
#                        result = self.process_sample(sample)
#
#                        if result is not None:
#                            original_img, cross_img, non_cross_img, label, _ = result  # Added non_cross_img
#                            original_list.append(original_img)
#                            cross_list.append(cross_img)
#                            non_cross_list.append(non_cross_img)  # Added
#                            labels_list.append(label)
#
#                    # Skip if no valid samples in batch
#                    if len(original_list) == 0:
#                        print(f"Skipping final metric calculation batch - all samples filtered due to empty non-cross regions")
#                        continue
#
#                    # Create batch tensors
#                    original_batch = torch.stack(original_list).to(self.device, non_blocking=True)
#                    cross_batch = torch.stack(cross_list).to(self.device, non_blocking=True)
#                    non_cross_batch = torch.stack(non_cross_list).to(self.device, non_blocking=True)  # Added
#                    label_batch = torch.stack(labels_list).to(self.device, non_blocking=True)
#
#                    # Model inference (pass three inputs)
#                    pred_logits, valid_mask = self.model(cross_batch, non_cross_batch, original_batch)  # Correct input order
#
#                    # Calculate precision, recall, F1 for this batch
#                    batch_precision, batch_recall, batch_f1 = self.calculate_precision_recall_f1(pred_logits,
#                                                                                                 label_batch)
#
#                    all_precisions.append(batch_precision)
#                    all_recalls.append(batch_recall)
#                    all_f1s.append(batch_f1)
#
#                    # Print progress
#                    if (batch_idx + 1) % 10 == 0 or batch_idx == len(self.val_loader) - 1:
#                        print(f"Metric evaluation progress: {batch_idx + 1}/{len(self.val_loader)} batches")
#
#                except Exception as e:
#                    print(f"Final metric evaluation batch {batch_idx} error: {e}")
#                    traceback.print_exc()
#                    continue
#
#        # Calculate average metrics across all batches
#        final_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0.0
#        final_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
#        final_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0
#
#        return final_precision, final_recall, final_f1
#
#    def prepare_data(self):
#        """Prepare data loaders and datasets (added non-cross region data)"""
#        print("Preparing data loaders")
#        start_time = time.time()
#
#        transform = transforms.Compose([
#            transforms.ToTensor(),  # Convert to tensor (labels processed separately)
#        ])
#
#        # Create training dataset
#        train_dataset = FastStrokeDataset(
#            self.args.data_dir,
#            transform=transform,
#            num_workers=self.args.num_workers,
#            preload=True
#        )
#        train_size = len(train_dataset)
#
#        # Create validation dataset (same format as training set, use test set as validation set)
#        val_dataset = FastStrokeDataset(
#            self.args.val_data_dir,
#            transform=transform,
#            num_workers=self.args.num_workers,
#            preload=True
#        )
#        val_size = len(val_dataset)
#
#        print(
#            f"Data preparation completed, training samples: {train_size}, validation samples: {val_size}, time taken: {time.time() - start_time:.2f}s")
#
#        # Create training data loader
#        try:
#            self.train_loader = DataLoader(
#                train_dataset,
#                batch_size=self.args.batch_size,
#                shuffle=True,
#                num_workers=self.args.num_workers,
#                pin_memory=True
#            )
#            print(
#                f"Training data loader setup completed, batch size: {self.args.batch_size}, threads: {self.args.num_workers}")
#        except Exception as e:
#            print(f"Training data loader setup failed: {e}")
#            raise
#
#        # Create validation data loader (half threads of training)
#        try:
#            self.val_loader = DataLoader(
#                val_dataset,
#                batch_size=self.args.batch_size,
#                shuffle=False,
#                num_workers=self.args.num_workers // 2,
#                pin_memory=True
#            )
#            print(
#                f"Validation data loader setup completed, batch size: {self.args.batch_size}, threads: {self.args.num_workers // 2}")
#        except Exception as e:
#            print(f"Validation data loader setup failed: {e}")
#            raise
#
#        print(f"Total data preparation time: {time.time() - start_time:.2f}s")
#        return self.train_loader, self.val_loader
#
#    def train(self):
#        """Main training loop (only use weighted BCE loss)"""
#        print("Starting SdaNet training (only use weighted BCE loss)")
#        start_epoch = 1 if not self.args.continue_training else self.args.start_epoch
#        print(f"Starting epoch: {start_epoch}")
#
#        total_epochs = self.args.epochs if not self.args.continue_training else self.args.start_epoch + self.args.num_epochs_continue - 1
#        print(f"Total training epochs: {total_epochs}")
#
#        for epoch in range(start_epoch, total_epochs + 1):
#            epoch_start = time.time()
#            print(f"Epoch {epoch}/{total_epochs} started")
#
#            current_lr = self.optimizer.param_groups[0]['lr']
#            print(f"Current learning rate: {current_lr:.8f}")
#            print(f"=" * 50)
#
#            # -------------------------- Training phase --------------------------
#            self.model.train()
#            train_total_loss = 0.0
#            train_bce_loss = 0.0
#            train_mean_iou = 0.0
#            batch_count = len(self.train_loader)
#            print(f"Starting training phase, total batches: {batch_count}")
#
#            for batch_idx, batch in enumerate(self.train_loader):
#                batch_start = time.time()
#
#                try:
#                    # Process batch samples
#                    original_list = []
#                    cross_list = []
#                    non_cross_list = []  # Added
#                    labels_list = []
#                    valid_samples = []
#
#                    for i in range(len(batch["original"])):
#                        sample = {
#                            "original": batch["original"][i],
#                            "cross": batch["cross"][i],  # Cross region image
#                            "non_cross": batch["non_cross"][i],  # Non-cross region image (added)
#                            "label": batch["label"][i],
#                            "folder_name": batch["folder_name"][i]
#                        }
#                        result = self.process_sample(sample)
#
#                        if result is not None:
#                            original_img, cross_img, non_cross_img, label, folder_name = result  # Added non_cross_img
#                            original_list.append(original_img)
#                            cross_list.append(cross_img)
#                            non_cross_list.append(non_cross_img)  # Added
#                            labels_list.append(label)
#                            valid_samples.append({
#                                "original": batch["original"][i],
#                                "cross": batch["cross"][i],
#                                "non_cross": batch["non_cross"][i],  # Added
#                                "label": batch["label"][i],
#                                "folder_name": folder_name
#                            })
#
#                    # Skip if no valid samples in batch
#                    if len(valid_samples) == 0:
#                        print(f"Skipping batch - all samples filtered due to empty non-cross regions")
#                        continue
#
#                    # Create batch tensors
#                    original_batch = torch.stack(original_list).to(self.device, non_blocking=True)
#                    cross_batch = torch.stack(cross_list).to(self.device, non_blocking=True)
#                    non_cross_batch = torch.stack(non_cross_list).to(self.device, non_blocking=True)  # Added
#                    label_batch = torch.stack(labels_list).to(self.device, non_blocking=True)
#
#                    # Print filtering info for first batch
#                    if batch_idx == 0:
#                        print(f"Sample filtering: {len(valid_samples)}/{len(batch['original'])} samples retained")
#
#                except Exception as e:
#                    print(f"Batch data processing failed: {e}")
#                    traceback.print_exc()
#                    raise
#
#                self.optimizer.zero_grad()
#
#                try:
#                    # Forward pass (pass three inputs, note order matches model)
#                    pred_logits, valid_mask = self.model(cross_batch, non_cross_batch, original_batch)
#
#                    # Calculate BCE loss
#                    total_loss, bce_loss_val, _ = self.combined_loss(pred_logits, label_batch)
#
#                    # Calculate IoU metric
#                    mean_iou, sample_ious = self.calculate_iou(pred_logits, label_batch)
#                    batch_mean_iou = sum(sample_ious) / len(sample_ious) if sample_ious else 0.0
#
#                except Exception as e:
#                    print(f"Forward pass failed: {e}")
#                    traceback.print_exc()
#                    raise
#
#                # Backward pass
#                try:
#                    total_loss.backward()
#                    
#                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#                    self.optimizer.step()
#                except Exception as e:
#                    print(f"Backward pass failed: {e}")
#                    traceback.print_exc()
#                    raise
#
#                # Accumulate loss and metrics
#                train_total_loss += total_loss.item()
#                train_bce_loss += bce_loss_val.item()
#                train_mean_iou += mean_iou.item()
#
#                # Batch progress
#                batch_time = time.time() - batch_start
#                if batch_idx % 10 == 0 or batch_idx == batch_count - 1:
#                    print(f"[Training] Batch {batch_idx + 1}/{batch_count} - "
#                          f"Total loss: {total_loss.item():.6f} - "
#                          f"Weighted BCE loss: {bce_loss_val.item():.6f} - "
#                          f"Mean IoU: {mean_iou.item():.6f} - "
#                          f"Learning rate: {current_lr:.8f} - Time taken: {batch_time:.4f}s")
#
#            # Update learning rate
#            self.scheduler.step()
#
#            # Calculate average training metrics
#            train_total_loss /= batch_count
#            train_bce_loss /= batch_count
#            train_mean_iou /= batch_count
#
#            print(f"Training phase completed - "
#                  f"Average total loss: {train_total_loss:.6f}, "
#                  f"Average weighted BCE loss: {train_bce_loss:.6f}, "
#                  f"Average IoU: {train_mean_iou:.6f}")
#
#            # -------------------------- Validation phase --------------------------
#            self.model.eval()
#            val_total_loss = 0.0
#            val_bce_loss = 0.0
#            val_mean_iou = 0.0
#            val_batch_count = len(self.val_loader)
#            print(f"Starting validation phase, total batches: {val_batch_count}")
#
#            with torch.no_grad():
#                for batch_idx, batch in enumerate(self.val_loader):
#                    batch_start = time.time()
#
#                    try:
#                        # Process batch samples
#                        original_list = []
#                        cross_list = []
#                        non_cross_list = []  # Added
#                        labels_list = []
#
#                        for i in range(len(batch["original"])):
#                            sample = {
#                                "original": batch["original"][i],
#                                "cross": batch["cross"][i],  # Cross region image
#                                "non_cross": batch["non_cross"][i],  # Non-cross region image (added)
#                                "label": batch["label"][i],
#                                "folder_name": batch["folder_name"][i]
#                            }
#                            result = self.process_sample(sample)
#                            if result is not None:  # Process filtered samples in validation set
#                                original_img, cross_img, non_cross_img, label, _ = result  # Added non_cross_img
#                                original_list.append(original_img)
#                                cross_list.append(cross_img)
#                                non_cross_list.append(non_cross_img)  # Added
#                                labels_list.append(label)
#
#                        # Skip if no valid samples in batch
#                        if len(original_list) == 0:
#                            print(f"Skipping validation batch - all samples filtered due to empty non-cross regions")
#                            continue
#
#                        # Create batch tensors
#                        original_batch = torch.stack(original_list).to(self.device, non_blocking=True)
#                        cross_batch = torch.stack(cross_list).to(self.device, non_blocking=True)
#                        non_cross_batch = torch.stack(non_cross_list).to(self.device, non_blocking=True)  # Added
#                        label_batch = torch.stack(labels_list).to(self.device, non_blocking=True)
#
#                    except Exception as e:
#                        print(f"Batch data processing failed: {e}")
#                        traceback.print_exc()
#                        raise
#
#                    try:
#                        # Model inference (pass three inputs)
#                        pred_logits, valid_mask = self.model(cross_batch, non_cross_batch, original_batch)
#
#                        # Calculate BCE loss
#                        total_loss_val, bce_loss_val, _ = self.combined_loss(pred_logits, label_batch)
#
#                        # Calculate validation IoU
#                        val_iou, val_sample_ious = self.calculate_iou(pred_logits, label_batch)
#                        val_batch_mean_iou = sum(val_sample_ious) / len(val_sample_ious) if val_sample_ious else 0.0
#
#                    except Exception as e:
#                        print(f"Forward pass failed: {e}")
#                        traceback.print_exc()
#                        raise
#
#                    # Accumulate validation metrics
#                    val_total_loss += total_loss_val.item()
#                    val_bce_loss += bce_loss_val.item()
#                    val_mean_iou += val_iou.item()
#
#                    # Batch progress
#                    batch_time = time.time() - batch_start
#                    if batch_idx % 10 == 0 or batch_idx == val_batch_count - 1:
#                        print(f"[Validation] Batch {batch_idx + 1}/{val_batch_count} - "
#                              f"Total loss: {total_loss_val.item():.6f} - "
#                              f"Weighted BCE loss: {bce_loss_val.item():.6f} - "
#                              f"Mean IoU: {val_iou.item():.6f} - Time taken: {batch_time:.4f}s")
#
#            # Calculate average validation metrics
#            val_total_loss /= val_batch_count
#            val_bce_loss /= val_batch_count
#            val_mean_iou /= val_batch_count
#
#            print(f"Validation phase completed - "
#                  f"Average total loss: {val_total_loss:.6f}, "
#                  f"Average weighted BCE loss: {val_bce_loss:.6f}, "
#                  f"Average IoU: {val_mean_iou:.6f}")
#
#            # -------------------------- Epoch summary --------------------------
#            epoch_time = time.time() - epoch_start
#            print(f"=" * 50)
#            print(f"Epoch {epoch}/{total_epochs} summary:")
#            print(
#                f"  Training - Total loss: {train_total_loss:.6f}, BCE: {train_bce_loss:.6f}, IoU: {train_mean_iou:.6f}")
#            print(
#                f"  Validation - Total loss: {val_total_loss:.6f}, BCE: {val_bce_loss:.6f}, IoU: {val_mean_iou:.6f}")
#            print(f"  Current learning rate: {current_lr:.8f}, Epoch time: {epoch_time:.2f}s")
#            print(f"=" * 50)
#
#            # Save visualization periodically
#            if epoch % self.args.visualize_frequency == 0:
#                self.visualize_epoch_predictions(epoch)
#
#            # Save best model (only save weights, no immediate visualization)
#            is_best = val_mean_iou > self.best_val_iou
#            if is_best:
#                self.best_val_iou = val_mean_iou
#                self.save_model(epoch, val_total_loss, val_mean_iou)
#                # Comment out visualization during training, execute after training
#                # self.save_visualizations()  # Do not generate best model visualization during training
#
#        # -------------------------- Post-training processing --------------------------
#        # 1. Calculate final metrics
#        print(f"\n{'=' * 60}")
#        print("Calculating final stroke extraction metrics on validation set...")
#        final_precision, final_recall, final_f1 = self.calculate_final_metrics()
#
#        # 2. Generate best model visualization (execute after training)
#        print(f"\n{'=' * 60}")
#        print("Training completed, starting best model visualization generation...")
#        self.save_visualizations()  # Unified visualization of best model after training
#
#        # 3. Save best model info with final metrics
#        self.save_best_info_with_metrics(final_precision, final_recall, final_f1)
#
#        # Training completion summary
#        print(f"{'=' * 60}")
#        print(f"Training completed, total time taken: {time.time() - self.start_time:.2f}s")
#        print(f"Best IoU: {self.best_val_iou:.6f}")
#        print(f"Final stroke extraction metrics:")
#        print(f"  Precision: {final_precision:.6f}")
#        print(f"  Recall: {final_recall:.6f}")
#        print(f"  F1 score: {final_f1:.6f}")
#        print(f"Best model saved to: {self.best_model_path}")
#        print(f"Best visualizations saved to: {self.best_vis_dir}")
#        print(f"Filtered samples (empty non-cross regions) saved to: {self.filtered_vis_dir}")
#        print(f"All epoch visualizations saved to: {self.args.model_save_dir}, folder names: epoch_*_visualization")
#        print(f"{'=' * 60}")
#
#    def save_model(self, epoch, val_loss, val_iou):
#        """Save best model checkpoint"""
#        try:
#            model_to_save = self._get_real_model()
#            torch.save(model_to_save.state_dict(), self.best_model_path)
#            print(f"Best model saved: {self.best_model_path} (Epoch: {epoch}, IoU: {val_iou:.6f})")
#        except Exception as e:
#            print(f"Model save failed: {e}")
#            raise
#
#    def save_best_info(self, epoch, val_loss, val_iou):
#        """Save best model info"""
#        with open(self.best_info_path, 'w') as f:
#            f.write("Best model info:\n")
#            f.write(f"Epoch: {epoch}\n")
#            f.write(f"Validation total loss: {val_loss:.6f}\n")
#            f.write(f"Validation mean IoU: {val_iou:.6f}\n")
#            f.write(f"Class weights: {self.class_weights.cpu().numpy()}\n")
#            f.write(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.8f}\n")
#            f.write(f"Embedding dimension: {self.args.embed_dim}\n")
#            f.write(f"Number of attention heads: {self.args.num_heads}\n")
#            f.write(f"Number of stroke classes: {self.args.num_stroke_classes}\n")
#            f.write(f"Batch size: {self.args.batch_size}\n")
#            f.write(f"Save time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
#            f.write(f"Image size: {self.target_size[0]}x{self.target_size[1]}\n")
#            f.write(f"Border color: {self.border_color}\n")
#            f.write(f"Border width: {self.border_width}\n")
#            f.write(f"Number of validation samples: {len(self.val_loader.dataset)} validation samples\n")
#        print(f"Best model info saved: {self.best_info_path}")
#
#    def save_best_info_with_metrics(self, final_precision, final_recall, final_f1):
#        """Save best model info with final metrics"""
#        with open(self.best_info_path, 'w') as f:
#            f.write("Best model info with final metrics:\n")
#            f.write(f"Best validation mean IoU: {self.best_val_iou:.6f}\n")
#            f.write(f"Final stroke extraction precision: {final_precision:.6f}\n")
#            f.write(f"Final stroke extraction recall: {final_recall:.6f}\n")
#            f.write(f"Final stroke extraction F1 score: {final_f1:.6f}\n")
#            f.write(f"Class weights: {self.class_weights.cpu().numpy()}\n")
#            f.write(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.8f}\n")
#            f.write(f"Embedding dimension: {self.args.embed_dim}\n")
#            f.write(f"Number of attention heads: {self.args.num_heads}\n")
#            f.write(f"Number of stroke classes: {self.args.num_stroke_classes}\n")
#            f.write(f"Batch size: {self.args.batch_size}\n")
#            f.write(f"Save time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
#            f.write(f"Image size: {self.target_size[0]}x{self.target_size[1]}\n")
#            f.write(f"Border color: {self.border_color}\n")
#            f.write(f"Border width: {self.border_width}\n")
#            f.write(f"Number of validation samples: {len(self.val_loader.dataset)} validation samples\n")
#            f.write(f"Filtered sample visualization: {self.filtered_vis_dir}\n")
#        print(f"Best model info with final metrics saved: {self.best_info_path}")
#
#
#def main():
#    """Main function to run training"""
#    parser = get_parser()
#    args = parser.parse_args()
#
#    # Validate parameters
#    if not os.path.exists(args.val_data_dir):
#        raise ValueError(f"Validation data directory does not exist: {args.val_data_dir}")
#    if args.continue_training and args.pretrained_model_path == "":
#        print(
#            "Error: --continue_training specified but --pretrained_model_path not provided, please provide SdaNet model path")
#        sys.exit(1)
#    # Validate class weight count
#    if len(args.class_weights.split(',')) != args.num_stroke_classes:
#        raise ValueError(
#            f"Number of class weights must match number of classes ({args.num_stroke_classes}), got {len(args.class_weights.split(','))}")
#
#    # Set GPU environment
#    args.gpu_ids = "0,1"
#    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
#    
#    if torch.cuda.is_available():
#        torch.cuda.empty_cache()
#        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
#
#    # Print training configuration
#    print(f"=" * 60)
#    print("SdaNet Training Configuration (only use weighted BCE loss)")
#    print(f"Python version: {sys.version.split()[0]}")
#    print(f"PyTorch version: {torch.__version__}")
#    print(f"CUDA version: {torch.version.cuda}")
#    print(f"Using GPU IDs: {args.gpu_ids}")
#    print(f"Model save directory: {args.model_save_dir}")
#    print(f"Class weights: {args.class_weights}")
#    print(f"Visualize filtered samples: {args.vis_filtered_samples}")
#    print(f"Save validation sample visualization every {args.visualize_frequency} epochs")
#    print(f"=" * 60)
#
#    # Start training
#    try:
#        trainer = SdaNetTrainer(args)
#        train_loader, val_loader = trainer.prepare_data()
#        trainer.train()
#    except Exception as e:
#        print(f"Training process failed: {str(e)}")
#        traceback.print_exc()
#        sys.exit(1)
#
#
#if __name__ == "__main__":
#    main()















## -*- coding: utf-8 -*-
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  
#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.utils.data import Dataset, DataLoader
#import torch.nn.functional as F
#import numpy as np
#import cv2
#
#import glob
#import time
#import sys
#import traceback
#import shutil
#from torchvision import transforms
#import argparse
#from concurrent.futures import ThreadPoolExecutor
#import torch.optim.lr_scheduler as lr_scheduler
#
#from model.SdaNet import SdaNet
#
#num_gpus = torch.cuda.device_count()
#use_data_parallel = num_gpus >= 2
#print(f"Detected {num_gpus} GPUs (Use DataParallel: {use_data_parallel})")
#
#
#def get_parser():
#    parser = argparse.ArgumentParser(description='SdaNet Training with Three Metrics and Spatial Weighting')
#
#    # Data parameters
#    parser.add_argument('--data_dir', type=str,
#                        default="/media/text/afedbba0-93b4-490d-9d12-20e5b7b22aa7/yan/exp2/S-Dauble-Attention/dataset/strock_semantics/semantics_kai/semantics_train",
#                        help='Training data directory')
#    parser.add_argument('--val_data_dir', type=str,
#                        default="/media/text/afedbba0-93b4-490d-9d12-20e5b7b22aa7/yan/exp2/S-Dauble-Attention/dataset/strock_semantics/semantics_kai/semantics_val",
#                        help='Validation data directory')
#    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
#    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading threads')
#
#    # Training parameters
#    parser.add_argument('--epochs', type=int, default=200, help='Total training epochs')
#    parser.add_argument('--initial_lr', type=float, default=0.00025, help='Initial learning rate')
#    parser.add_argument('--num_stroke_classes', type=int, default=8, help='Number of stroke classes')
#
#    # Loss function parameters
#    parser.add_argument('--class_weights', type=str,
#                        default="1.35,1.25,1.25,1.0,1.0,1.25,1.25,1.35",
#                        help='Class weights (comma-separated)')
#
#    # Model parameters
#    parser.add_argument('--model_save_dir', type=str,
#                        default="/media/text/afedbba0-93b4-490d-9d12-20e5b7b22aa7/yan/exp2/S-Dauble-Attention/dataset/weight/kai3",
#                        help='Model save directory')
#    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
#    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
#
#    # GPU settings
#    parser.add_argument('--gpu_ids', type=str, default="0,1", help='GPU IDs to use, e.g., "0,1"')
#
#    # Resume training
#    parser.add_argument('--continue_training', action='store_true', default=False,
#                        help='Resume training from checkpoint')
#    parser.add_argument('--pretrained_model_path', type=str, default="", help='Pretrained model path')
#    parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch')
#    parser.add_argument('--num_epochs_continue', type=int, default=100, help='Number of epochs to resume')
#
#    # Loss function type
#    parser.add_argument('--loss_type', type=str, default='weighted_bce', choices=['weighted_bce'],
#                        help='Loss function type')
#
#    # Visualisation parameters
#    parser.add_argument('--visualize_frequency', type=int, default=99, help='Visualisation save frequency (epochs)')
#    parser.add_argument('--target_size', type=int, default=1024, help='Visualisation image target size')
#    parser.add_argument('--border_width', type=int, default=64, help='Border width')
#    parser.add_argument('--vis_filtered_samples', action='store_true', default=True,
#                        help='Visualize samples filtered due to empty non-cross regions')
#
#    return parser
#
#
#class FastStrokeDataset(Dataset):
#    def __init__(self, data_dir, transform=None, num_workers=8, preload=False):
#        self.data_dir = data_dir
#        self.transform = transform
#        self.num_workers = num_workers
#        self.preload = preload
#
#        if not os.path.exists(data_dir):
#            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
#
#        self.image_folders = [f for f in glob.glob(os.path.join(data_dir, "image_*")) if os.path.isdir(f)]
#        self.image_folders.sort()
#        print(f"Found {len(self.image_folders)} image folders in {data_dir}")
#
#        self.data = []
#        for i, folder in enumerate(self.image_folders):
#            folder_name = os.path.basename(folder)
#            try:
#                original_files = glob.glob(os.path.join(folder, "*union*"))
#                cross_files = glob.glob(os.path.join(folder, "*intersection*"))
#                non_cross_files = glob.glob(os.path.join(folder, "*processed*"))
#                label_files = glob.glob(os.path.join(folder, "*label.npy"))
#
#                if original_files and cross_files and non_cross_files and label_files:
#                    self.data.append({
#                        "original": original_files[0],
#                        "cross": cross_files[0],
#                        "non_cross": non_cross_files[0],
#                        "label": label_files[0],
#                        "folder_name": folder_name
#                    })
#                    if i < 5 or i % 1000 == 0:
#                        print(f"Loading data, {i}/{len(self.image_folders)}: {folder_name}")
#                else:
#                    missing = []
#                    if not original_files:
#                        missing.append("union.png (original image)")
#                    if not cross_files:
#                        missing.append("_intersection.png (cross region image)")
#                    if not non_cross_files:
#                        missing.append("_final_processed.png (non-cross region image)")
#                    if not label_files:
#                        missing.append("label.npy (label)")
#                    print(f"Warning: Folder {folder_name} missing files: {missing}, skipped")
#            except Exception as e:
#                print(f"Error processing folder {folder_name}: {e}")
#
#        print(f"Total samples in {data_dir} dataset: {len(self.data)}")
#
#        if preload:
#            self._preload_data()
#
#    def _preload_data(self):
#        print(f"Preloading data with {self.num_workers} threads...")
#        start_time = time.time()
#        try:
#            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
#                self.preloaded_data = list(executor.map(lambda x: self.data[x], range(len(self.data))))
#        except Exception as e:
#            print(f"Data preloading error: {e}")
#            traceback.print_exc()
#            raise
#        end_time = time.time()
#        print(f"Data preloading completed, {len(self.preloaded_data)} samples, time taken: {end_time - start_time:.2f}s")
#
#    def __len__(self):
#        return len(self.data) if not self.preload else len(self.preloaded_data)
#
#    def __getitem__(self, idx):
#        if self.preload:
#            return self.preloaded_data[idx]
#        return self.data[idx]
#
#
#class SdaNetTrainer:
#    def __init__(self, args):
#        self.args = args
#        self.best_val_iou = -float('inf')
#        self.best_model_path = os.path.join(args.model_save_dir, "best_model.pth")
#        self.best_vis_dir = os.path.join(args.model_save_dir, "best_visualization")
#        self.best_info_path = os.path.join(args.model_save_dir, "best_info.txt")
#        self.start_time = time.time()
#
#        self.filtered_vis_dir = os.path.join(args.model_save_dir,
#                                             f"filtered_samples_vis_{time.strftime('%Y%m%d_%H%M%S')}")
#        if args.vis_filtered_samples:
#            os.makedirs(self.filtered_vis_dir, exist_ok=True)
#            print(f"Filtered samples (empty non-cross regions) will be saved to: {self.filtered_vis_dir}")
#
#        self.class_weights = torch.tensor([float(w) for w in args.class_weights.split(',')], dtype=torch.float32)
#        assert len(self.class_weights) == args.num_stroke_classes
#        print(f"Class weights: {self.class_weights.numpy()}")
#
#        self.target_size = (args.target_size, args.target_size)
#        self.border_color = (230, 230, 10)
#        self.border_width = args.border_width
#        np.random.seed(42)
#        self.colors = np.random.randint(50, 256, size=(args.num_stroke_classes, 3), dtype=np.uint8)
#        print(f"Generated {args.num_stroke_classes} colors, target size: {self.target_size[0]}x{self.target_size[1]}")
#
#        print("Trainer initialization completed")
#
#        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#        self.class_weights = self.class_weights.to(self.device)
#        print(f"Using device: {self.device}")
#
#        self.num_gpus = torch.cuda.device_count()
#        if self.num_gpus > 1:
#            print(f"Detected {self.num_gpus} GPUs, using DataParallel for multi-GPU training")
#        else:
#            print(f"Detected {self.num_gpus} GPU, using single GPU training")
#
#        os.makedirs(args.model_save_dir, exist_ok=True)
#        print(f"Model save directory: {args.model_save_dir}")
#
#        print(f"Initializing SdaNet")
#        self.model = SdaNet(
#            embed_dim=args.embed_dim,
#            num_heads=args.num_heads,
#            num_stroke_classes=args.num_stroke_classes,
#        ).to(self.device)
#        print("Model initialization completed")
#
#        try:
#            if torch.cuda.is_available():
#                torch.cuda.empty_cache()
#                for i in range(self.num_gpus):
#                    mem_info = torch.cuda.mem_get_info(i)
#                    print(f"GPU {i} memory: Free {mem_info[0] / 1024 ** 3:.2f} GB, Total {mem_info[1] / 1024 ** 3:.2f} GB")
#        except Exception as e:
#            print(f"Memory check error: {e}")
#            raise
#
#        if self.num_gpus > 1:
#            try:
#                self.model = nn.DataParallel(self.model)
#                print(f"DataParallel setup successful, using {self.num_gpus} GPUs")
#            except Exception as e:
#                print(f"DataParallel setup failed: {e}")
#                raise
#
#        self.optimizer = optim.Adam(self.model.parameters(), lr=args.initial_lr,
#                                    betas=(0.9, 0.999), weight_decay=1e-5)
#        print("Optimizer setup completed (Adam)")
#
#        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=60, eta_min=1e-6)
#        print(f"Learning rate scheduler setup completed, time taken: {time.time() - self.start_time:.2f}s")
#
#    def _get_real_model(self):
#        if isinstance(self.model, torch.nn.DataParallel):
#            return self.model.module
#        return self.model
#
#    # ---------- Three?metric IoU calculation ----------
#    def calculate_three_miou(self, pred_logits, label, eps=1e-6):
#        pred = (torch.sigmoid(pred_logits) > 0.5).float()
#        B, C, H, W = pred.shape
#    
#        total_list, cross_list, noncross_list = [], [], []
#    
#        for b in range(B):
#            p = pred[b]
#            t = label[b]
#            sum_t = t.sum(dim=0)
#    
#            cross_mask = (sum_t >= 2).float()
#            noncross_mask = (sum_t <= 1).float()
#    
#            # Total mIoU
#            t_ious = []
#            for c in range(C):
#                inter = (p[c] * t[c]).sum()
#                union = (p[c] + t[c]).sum() - inter
#               
#                iou = ((inter + eps) / (union + eps)).item()
#                t_ious.append(iou)
#            total_list.append(sum(t_ious) / C)
#    
#            # Cross mIoU
#            c_ious = []
#            for c in range(C):
#                inter = (p[c] * t[c] * cross_mask).sum()
#                union = ((p[c] + t[c]) * cross_mask).sum() - inter
#                if union > 0:
#                    iou = ((inter + eps) / (union + eps)).item()
#                    c_ious.append(iou)
#            cross_list.append(sum(c_ious) / len(c_ious) if c_ious else 1.0)
#    
#            # Non-Cross mIoU
#            n_ious = []
#            for c in range(C):
#                inter = (p[c] * t[c] * noncross_mask).sum()
#                union = ((p[c] + t[c]) * noncross_mask).sum() - inter
#                if union > 0:
#                    iou = ((inter + eps) / (union + eps)).item()
#                    n_ious.append(iou)
#            noncross_list.append(sum(n_ious) / len(n_ious) if n_ious else 1.0)
#
#   
#        return np.mean(total_list), np.mean(cross_list), np.mean(noncross_list)
#
#    # ---------- Loss functions with spatial weighting for non-cross pixels ----------
#    def weighted_bce_loss(self, pred_logits, target):
#        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
#        class_weights = self.class_weights.view(1, -1, 1, 1)
#
#        # Spatial weight: give higher weight to non-cross pixels (sum of classes <= 1)
#        sum_target = target.sum(dim=1, keepdim=True)          # [B, 1, H, W]
#        spatial_weight = torch.where(sum_target <= 1,
#                                     torch.tensor(1.5, device=target.device),
#                                     torch.tensor(1.0, device=target.device))
#
#        weighted_bce = bce * class_weights * spatial_weight
#        return weighted_bce.mean()
#
#    def combined_loss(self, pred_logits, target):
#        bce = self.weighted_bce_loss(pred_logits, target)
#        pred = torch.sigmoid(pred_logits)
#
#        inter = (pred * target).sum(dim=(2, 3))
#        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
#        dice = 1 - (2 * inter + 1e-5) / (union + 1e-5)
#        dice = dice.mean()
#
#        total_loss = 0.5 * bce + 0.5 * dice
#        return total_loss, bce, dice
#
#    # ---------- Sample processing with non-cross enhancement (training only) ----------
#    def process_sample(self, sample, is_train=True):
#        original_img = cv2.imread(sample["original"], cv2.IMREAD_GRAYSCALE)
#        cross_img = cv2.imread(sample["cross"], cv2.IMREAD_GRAYSCALE)
#        non_cross_img = cv2.imread(sample["non_cross"], cv2.IMREAD_GRAYSCALE)
#        label = np.load(sample["label"])
#        folder_name = sample["folder_name"]
#
#        if original_img is None:
#            raise ValueError(f"Failed to load original image, folder: {folder_name}, path: {sample['original']}")
#        if cross_img is None:
#            raise ValueError(f"Failed to load cross region image, folder: {folder_name}, path: {sample['cross']}")
#        if non_cross_img is None:
#            raise ValueError(f"Failed to load non-cross region image, folder: {folder_name}, path: {sample['non_cross']}")
#
#        if original_img.shape != (128, 128):
#            original_img = cv2.resize(original_img, (128, 128), interpolation=cv2.INTER_NEAREST)
#        if cross_img.shape != (128, 128):
#            cross_img = cv2.resize(cross_img, (128, 128), interpolation=cv2.INTER_NEAREST)
#        if non_cross_img.shape != (128, 128):
#            non_cross_img = cv2.resize(non_cross_img, (128, 128), interpolation=cv2.INTER_NEAREST)
#
#        if label.shape[0] != 128 or label.shape[1] != 128:
#            label_resized = np.zeros((128, 128, label.shape[2]), dtype=label.dtype)
#            for c in range(label.shape[2]):
#                label_resized[:, :, c] = cv2.resize(label[:, :, c], (128, 128), interpolation=cv2.INTER_NEAREST)
#            label = label_resized
#
#        original_tensor = torch.from_numpy(original_img).float().unsqueeze(0) / 255.0
#        cross_tensor = torch.from_numpy(cross_img).float().unsqueeze(0) / 255.0
#        non_cross_tensor = torch.from_numpy(non_cross_img).float().unsqueeze(0) / 255.0
#
#        if is_train:
#            # Reduced enhancement factor + small Gaussian noise for generalisation
#            non_cross_tensor = non_cross_tensor * 1.15
#            non_cross_tensor = non_cross_tensor + torch.randn_like(non_cross_tensor) * 0.01
#            non_cross_tensor = torch.clamp(non_cross_tensor, 0, 1)
#
#        label_tensor = torch.from_numpy(label).float().permute(2, 0, 1)
#        return original_tensor, cross_tensor, non_cross_tensor, label_tensor, folder_name
#
#    # ---------- Visualisation functions (kept from original) ----------
#    def get_color_image(self, single_image, white_bg=True, apply_threshold=True):
#        if isinstance(single_image, torch.Tensor):
#            single_image = single_image.detach().cpu().numpy()
#        if apply_threshold and len(single_image.shape) == 3:
#            single_image = (torch.sigmoid(torch.from_numpy(single_image)) > 0.5).numpy()
#        C, H, W = single_image.shape
#        color_img = np.full((H, W, 3), 255, dtype=np.uint8) if white_bg else np.zeros((H, W, 3), dtype=np.uint8)
#        for c in range(C):
#            if c >= len(self.colors):
#                continue
#            mask = single_image[c] > 0.5
#            color = self.colors[c]
#            color_img[mask] = np.clip(color_img[mask] + color, 0, 255)
#        return color_img
#
#    def add_border(self, img, color, width):
#        h, w = img.shape[:2]
#        return cv2.rectangle(img.copy(), (0, 0), (w - 1, h - 1), color, thickness=width)
#
#    def visualize_epoch_predictions(self, epoch):
#        epoch_vis_dir = os.path.join(self.args.model_save_dir, f"epoch_{epoch}_visualization")
#        if os.path.exists(epoch_vis_dir):
#            shutil.rmtree(epoch_vis_dir)
#        os.makedirs(epoch_vis_dir, exist_ok=True)
#
#        test_dataset = self.val_loader.dataset
#        total_samples = len(test_dataset)
#        print(f"\n[Visualisation] Starting epoch {epoch} visualisation, processing {total_samples} samples, save path: {epoch_vis_dir}")
#
#        self.model.eval()
#        with torch.no_grad():
#            for vis_idx, data_idx in enumerate(range(total_samples)):
#                try:
#                    sample = test_dataset[data_idx]
#                    result = self.process_sample(sample, is_train=False)  # No enhancement for visualisation
#                    if result is None:
#                        continue
#                    original_img, cross_img, non_cross_img, label, folder_name = result
#
#                    original_batch = original_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    cross_batch = cross_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    non_cross_batch = non_cross_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    pred_logits, _ = self.model(cross_batch, non_cross_batch, original_batch)
#                    pred_prob = torch.sigmoid(pred_logits)
#
#                    original_np = original_img.squeeze().cpu().numpy()
#                    original_rgb = cv2.cvtColor((original_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    original_rgb_target = cv2.resize(original_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    cross_np = cross_img.squeeze().cpu().numpy()
#                    cross_rgb = cv2.cvtColor((cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    cross_rgb_target = cv2.resize(cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    non_cross_np = non_cross_img.squeeze().cpu().numpy()
#                    non_cross_rgb = cv2.cvtColor((non_cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    non_cross_rgb_target = cv2.resize(non_cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    label_np = label.cpu().numpy()
#                    label_color = self.get_color_image(label_np, white_bg=True, apply_threshold=False)
#                    label_color_target = cv2.resize(label_color, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    pred_np = pred_prob.squeeze(0).cpu().numpy()
#                    pred_color = self.get_color_image(pred_np, white_bg=True, apply_threshold=True)
#                    pred_color_target = cv2.resize(pred_color, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    original_with_border = self.add_border(original_rgb_target, self.border_color, self.border_width)
#                    cross_with_border = self.add_border(cross_rgb_target, self.border_color, self.border_width)
#                    non_cross_with_border = self.add_border(non_cross_rgb_target, self.border_color, self.border_width)
#                    label_with_border = self.add_border(label_color_target, self.border_color, self.border_width)
#                    prediction_with_border = self.add_border(pred_color_target, self.border_color, self.border_width)
#
#                    save_paths = {
#                        "original": os.path.join(epoch_vis_dir, f"{folder_name}_original_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "cross": os.path.join(epoch_vis_dir, f"{folder_name}_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "non_cross": os.path.join(epoch_vis_dir, f"{folder_name}_non_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "label": os.path.join(epoch_vis_dir, f"{folder_name}_label_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "prediction": os.path.join(epoch_vis_dir, f"{folder_name}_prediction_{self.target_size[0]}x{self.target_size[1]}.png")
#                    }
#                    for name, path in save_paths.items():
#                        cv2.imwrite(path, locals()[name + "_with_border"])
#
#                    if (vis_idx + 1) % 10 == 0 or vis_idx == total_samples - 1:
#                        print(f"[Visualisation] Epoch {epoch} progress: {vis_idx + 1}/{total_samples} samples, current folder: {folder_name}")
#                except Exception as e:
#                    print(f"[Visualisation Error] Epoch {epoch} sample {folder_name} failed: {str(e)}")
#                    continue
#        print(f"[Visualisation] Epoch {epoch} visualisation completed, saved to: {epoch_vis_dir}\n")
#
#    def save_visualizations(self):
#        if os.path.exists(self.best_vis_dir):
#            shutil.rmtree(self.best_vis_dir)
#        os.makedirs(self.best_vis_dir, exist_ok=True)
#
#        try:
#            model_to_load = self._get_real_model()
#            model_to_load.load_state_dict(torch.load(self.best_model_path))
#            print(f"Loaded best model weights for visualisation: {self.best_model_path}")
#        except Exception as e:
#            print(f"Failed to load best model weights, using current model state for visualisation: {e}")
#
#        test_dataset = self.val_loader.dataset
#        total_samples = len(test_dataset)
#        print(f"\n[Best Model Visualisation] Generating visualisations for {total_samples} validation samples, save path: {self.best_vis_dir}")
#
#        self.model.eval()
#        with torch.no_grad():
#            for vis_idx, data_idx in enumerate(range(total_samples)):
#                try:
#                    sample = test_dataset[data_idx]
#                    result = self.process_sample(sample, is_train=False)
#                    if result is None:
#                        continue
#                    original_img, cross_img, non_cross_img, label, folder_name = result
#
#                    original_batch = original_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    cross_batch = cross_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    non_cross_batch = non_cross_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    pred_logits, _ = self.model(cross_batch, non_cross_batch, original_batch)
#                    pred_prob = torch.sigmoid(pred_logits)
#
#                    original_np = original_img.squeeze().cpu().numpy()
#                    original_rgb = cv2.cvtColor((original_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    original_rgb_target = cv2.resize(original_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    cross_np = cross_img.squeeze().cpu().numpy()
#                    cross_rgb = cv2.cvtColor((cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    cross_rgb_target = cv2.resize(cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    non_cross_np = non_cross_img.squeeze().cpu().numpy()
#                    non_cross_rgb = cv2.cvtColor((non_cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    non_cross_rgb_target = cv2.resize(non_cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    label_np = label.cpu().numpy()
#                    label_color = self.get_color_image(label_np, white_bg=True, apply_threshold=False)
#                    label_color_target = cv2.resize(label_color, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    pred_np = pred_prob.squeeze(0).cpu().numpy()
#                    pred_color = self.get_color_image(pred_np, white_bg=True, apply_threshold=False)
#                    pred_color_target = cv2.resize(pred_color, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    original_with_border = self.add_border(original_rgb_target, self.border_color, self.border_width)
#                    cross_with_border = self.add_border(cross_rgb_target, self.border_color, self.border_width)
#                    non_cross_with_border = self.add_border(non_cross_rgb_target, self.border_color, self.border_width)
#                    label_with_border = self.add_border(label_color_target, self.border_color, self.border_width)
#                    prediction_with_border = self.add_border(pred_color_target, self.border_color, self.border_width)
#
#                    save_paths = {
#                        "original": os.path.join(self.best_vis_dir, f"{folder_name}_original_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "cross": os.path.join(self.best_vis_dir, f"{folder_name}_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "non_cross": os.path.join(self.best_vis_dir, f"{folder_name}_non_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "label": os.path.join(self.best_vis_dir, f"{folder_name}_label_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "prediction": os.path.join(self.best_vis_dir, f"{folder_name}_prediction_{self.target_size[0]}x{self.target_size[1]}.png")
#                    }
#                    for name, path in save_paths.items():
#                        cv2.imwrite(path, locals()[name + "_with_border"])
#
#                    if (vis_idx + 1) % 10 == 0 or vis_idx == total_samples - 1:
#                        print(f"[Best Model Visualisation] Progress: {vis_idx + 1}/{total_samples} samples, current folder: {folder_name}")
#                except Exception as e:
#                    print(f"[Best Model Visualisation Error] Sample {folder_name} failed: {str(e)}")
#                    continue
#        print(f"[Best Model Visualisation] All visualisations completed, saved to: {self.best_vis_dir}\n")
#
#    # ---------- Data preparation ----------
#    def prepare_data(self):
#        print("Preparing data loaders")
#        start_time = time.time()
#        transform = transforms.Compose([transforms.ToTensor()])
#
#        train_dataset = FastStrokeDataset(self.args.data_dir, transform=transform,
#                                          num_workers=self.args.num_workers, preload=True)
#        val_dataset = FastStrokeDataset(self.args.val_data_dir, transform=transform,
#                                        num_workers=self.args.num_workers, preload=True)
#
#        train_size = len(train_dataset)
#        val_size = len(val_dataset)
#        print(f"Data preparation completed, training samples: {train_size}, validation samples: {val_size}, time taken: {time.time() - start_time:.2f}s")
#
#        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True,
#                                       num_workers=self.args.num_workers, pin_memory=True)
#        self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False,
#                                     num_workers=self.args.num_workers // 2, pin_memory=True)
#        print(f"Training data loader setup completed, batch size: {self.args.batch_size}, threads: {self.args.num_workers}")
#        print(f"Validation data loader setup completed, batch size: {self.args.batch_size}, threads: {self.args.num_workers // 2}")
#        print(f"Total data preparation time: {time.time() - start_time:.2f}s")
#        return self.train_loader, self.val_loader
#
#    # ---------- Main training loop with three metrics ----------
#    def train(self):
#        print("Starting SdaNet training with three metrics and spatial weighting")
#        start_epoch = 1 if not self.args.continue_training else self.args.start_epoch
#        total_epochs = self.args.epochs if not self.args.continue_training else self.args.start_epoch + self.args.num_epochs_continue - 1
#        print(f"Starting epoch: {start_epoch}, Total training epochs: {total_epochs}")
#
#        for epoch in range(start_epoch, total_epochs + 1):
#            epoch_start = time.time()
#            current_lr = self.optimizer.param_groups[0]['lr']
#            print(f"\nEpoch {epoch}/{total_epochs} started, LR={current_lr:.8f}")
#            print("=" * 60)
#
#            # ---------- Training phase ----------
#            self.model.train()
#            train_total_loss = 0.0
#            train_bce_loss = 0.0
#            train_total_miou, train_cross_miou, train_nc_miou = 0.0, 0.0, 0.0
#            batch_count = len(self.train_loader)
#
#            for batch_idx, batch in enumerate(self.train_loader):
#                orig_list, cross_list, noncross_list, label_list = [], [], [], []
#                for i in range(len(batch["original"])):
#                    sample = {
#                        "original": batch["original"][i],
#                        "cross": batch["cross"][i],
#                        "non_cross": batch["non_cross"][i],
#                        "label": batch["label"][i],
#                        "folder_name": batch["folder_name"][i]
#                    }
#                    result = self.process_sample(sample, is_train=True)
#                    if result is None:
#                        continue
#                    o, c, n, l, _ = result
#                    orig_list.append(o)
#                    cross_list.append(c)
#                    noncross_list.append(n)
#                    label_list.append(l)
#
#                if len(orig_list) == 0:
#                    continue
#
#                orig_batch = torch.stack(orig_list).to(self.device)
#                cross_batch = torch.stack(cross_list).to(self.device)
#                noncross_batch = torch.stack(noncross_list).to(self.device)
#                label_batch = torch.stack(label_list).to(self.device)
#
#                self.optimizer.zero_grad()
#                pred_logits, _ = self.model(cross_batch, noncross_batch, orig_batch)
#                total_loss, bce_loss, _ = self.combined_loss(pred_logits, label_batch)
#                total_miou, cross_miou, nc_miou = self.calculate_three_miou(pred_logits, label_batch)
#
#                total_loss.backward()
#                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#                self.optimizer.step()
#
#                train_total_loss += total_loss.item()
#                train_bce_loss += bce_loss.item()
#                train_total_miou += total_miou
#                train_cross_miou += cross_miou
#                train_nc_miou += nc_miou
#
#                if batch_idx % 10 == 0 or batch_idx == batch_count - 1:
#                    print(f"[Train Batch {batch_idx+1}/{batch_count}] Loss: {total_loss.item():.4f}, "
#                          f"Total mIoU: {total_miou:.4f}, Cross: {cross_miou:.4f}, NonCross: {nc_miou:.4f}")
#
#            train_total_loss /= batch_count
#            train_bce_loss /= batch_count
#            train_total_miou /= batch_count
#            train_cross_miou /= batch_count
#            train_nc_miou /= batch_count
#
#            print(f"[Train] Avg Loss: {train_total_loss:.4f}, BCE: {train_bce_loss:.4f}, "
#                  f"Total mIoU: {train_total_miou:.4f}, Cross: {train_cross_miou:.4f}, NonCross: {train_nc_miou:.4f}")
#
#            # ---------- Validation phase ----------
#            self.model.eval()
#            val_total_loss = 0.0
#            val_bce_loss = 0.0
#            val_total_miou, val_cross_miou, val_nc_miou = 0.0, 0.0, 0.0
#            val_batch_count = len(self.val_loader)
#
#            with torch.no_grad():
#                for batch_idx, batch in enumerate(self.val_loader):
#                    orig_list, cross_list, noncross_list, label_list = [], [], [], []
#                    for i in range(len(batch["original"])):
#                        sample = {
#                            "original": batch["original"][i],
#                            "cross": batch["cross"][i],
#                            "non_cross": batch["non_cross"][i],
#                            "label": batch["label"][i],
#                            "folder_name": batch["folder_name"][i]
#                        }
#                        result = self.process_sample(sample, is_train=False)  # No enhancement
#                        if result is None:
#                            continue
#                        o, c, n, l, _ = result
#                        orig_list.append(o)
#                        cross_list.append(c)
#                        noncross_list.append(n)
#                        label_list.append(l)
#
#                    if len(orig_list) == 0:
#                        continue
#
#                    orig_batch = torch.stack(orig_list).to(self.device)
#                    cross_batch = torch.stack(cross_list).to(self.device)
#                    noncross_batch = torch.stack(noncross_list).to(self.device)
#                    label_batch = torch.stack(label_list).to(self.device)
#
#                    pred_logits, _ = self.model(cross_batch, noncross_batch, orig_batch)
#                    total_loss, bce_loss, _ = self.combined_loss(pred_logits, label_batch)
#                    total_miou, cross_miou, nc_miou = self.calculate_three_miou(pred_logits, label_batch)
#
#                    val_total_loss += total_loss.item()
#                    val_bce_loss += bce_loss.item()
#                    val_total_miou += total_miou
#                    val_cross_miou += cross_miou
#                    val_nc_miou += nc_miou
#
#                    if batch_idx % 10 == 0 or batch_idx == val_batch_count - 1:
#                        print(f"[Val Batch {batch_idx+1}/{val_batch_count}] Loss: {total_loss.item():.4f}, "
#                              f"Total mIoU: {total_miou:.4f}, Cross: {cross_miou:.4f}, NonCross: {nc_miou:.4f}")
#
#            val_total_loss /= val_batch_count
#            val_bce_loss /= val_batch_count
#            val_total_miou /= val_batch_count
#            val_cross_miou /= val_batch_count
#            val_nc_miou /= val_batch_count
#
#            print(f"[Val] Avg Loss: {val_total_loss:.4f}, BCE: {val_bce_loss:.4f}, "
#                  f"Total mIoU: {val_total_miou:.4f}, Cross: {val_cross_miou:.4f}, NonCross: {val_nc_miou:.4f}")
#
#            self.scheduler.step()
#
#            # Save best model based on Total mIoU
#            if val_total_miou > self.best_val_iou:
#                self.best_val_iou = val_total_miou
#                self.save_model(epoch, val_total_loss, val_total_miou)
#                self.save_best_info_with_metrics(val_total_miou, val_cross_miou, val_nc_miou)
#
#            # Visualisation
#            if epoch % self.args.visualize_frequency == 0:
#                self.visualize_epoch_predictions(epoch)
#
#            epoch_time = time.time() - epoch_start
#            print(f"Epoch {epoch} finished in {epoch_time:.2f}s\n")
#
#        # Post-training
#        print("\n" + "="*60)
#        print("Training completed, generating best model visualisations...")
#        self.save_visualizations()
#        print(f"Total training time: {time.time() - self.start_time:.2f}s")
#        print(f"Best Total mIoU: {self.best_val_iou:.6f}")
#        print("="*60)
#
#    def save_model(self, epoch, val_loss, val_iou):
#        try:
#            model_to_save = self._get_real_model()
#            torch.save(model_to_save.state_dict(), self.best_model_path)
#            print(f"Best model saved: {self.best_model_path} (Epoch: {epoch}, Total mIoU: {val_iou:.6f})")
#        except Exception as e:
#            print(f"Model save failed: {e}")
#            raise
#
#    def save_best_info_with_metrics(self, total_miou, cross_miou, noncross_miou):
#        with open(self.best_info_path, 'w') as f:
#            f.write("Best model info with three metrics:\n")
#            f.write(f"Best validation Total mIoU: {total_miou:.6f}\n")
#            f.write(f"Best validation Cross mIoU: {cross_miou:.6f}\n")
#            f.write(f"Best validation Non-Cross mIoU: {noncross_miou:.6f}\n")
#            f.write(f"Class weights: {self.class_weights.cpu().numpy()}\n")
#            f.write(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.8f}\n")
#            f.write(f"Embedding dimension: {self.args.embed_dim}\n")
#            f.write(f"Number of attention heads: {self.args.num_heads}\n")
#            f.write(f"Number of stroke classes: {self.args.num_stroke_classes}\n")
#            f.write(f"Batch size: {self.args.batch_size}\n")
#            f.write(f"Save time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
#            f.write(f"Image size: {self.target_size[0]}x{self.target_size[1]}\n")
#            f.write(f"Border color: {self.border_color}\n")
#            f.write(f"Border width: {self.border_width}\n")
#            f.write(f"Number of validation samples: {len(self.val_loader.dataset)}\n")
#            f.write(f"Filtered sample visualisation: {self.filtered_vis_dir}\n")
#        print(f"Best model info saved: {self.best_info_path}")
#
#
#def main():
#    parser = get_parser()
#    args = parser.parse_args()
#
#    if not os.path.exists(args.val_data_dir):
#        raise ValueError(f"Validation data directory does not exist: {args.val_data_dir}")
#    if args.continue_training and args.pretrained_model_path == "":
#        print("Error: --continue_training specified but --pretrained_model_path not provided")
#        sys.exit(1)
#    if len(args.class_weights.split(',')) != args.num_stroke_classes:
#        raise ValueError(f"Number of class weights must match number of classes ({args.num_stroke_classes})")
#
#    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
#    if torch.cuda.is_available():
#        torch.cuda.empty_cache()
#        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
#
#    print("="*60)
#    print("SdaNet Training Configuration (Three Metrics + Spatial Weighting)")
#    print(f"Python version: {sys.version.split()[0]}")
#    print(f"PyTorch version: {torch.__version__}")
#    print(f"CUDA version: {torch.version.cuda}")
#    print(f"Using GPU IDs: {args.gpu_ids}")
#    print(f"Model save directory: {args.model_save_dir}")
#    print(f"Class weights: {args.class_weights}")
#    print(f"Visualize filtered samples: {args.vis_filtered_samples}")
#    print(f"Save validation sample visualisation every {args.visualize_frequency} epochs")
#    print("="*60)
#
#    try:
#        trainer = SdaNetTrainer(args)
#        trainer.prepare_data()
#        trainer.train()
#    except Exception as e:
#        print(f"Training process failed: {str(e)}")
#        traceback.print_exc()
#        sys.exit(1)
#
#
#if __name__ == "__main__":
#    main()







#
#
##final 
# -*- coding: utf-8 -*-
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  
#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.utils.data import Dataset, DataLoader
#import torch.nn.functional as F
#import numpy as np
#import cv2
#
#import glob
#import time
#import sys
#import traceback
#import shutil
#from torchvision import transforms
#import argparse
#from concurrent.futures import ThreadPoolExecutor
#import torch.optim.lr_scheduler as lr_scheduler
#
#from model.SdaNet import SdaNet
#
#num_gpus = torch.cuda.device_count()
#use_data_parallel = num_gpus >= 2
#print(f"Detected {num_gpus} GPUs (Use DataParallel: {use_data_parallel})")
#
#
#def get_parser():
#    parser = argparse.ArgumentParser(description='SdaNet Training with Three Metrics and Spatial Weighting')
#
#    # Data parameters
#    parser.add_argument('--data_dir', type=str,
#                        default="/media/text/afedbba0-93b4-490d-9d12-20e5b7b22aa7/yan/exp2/S-Dauble-Attention/dataset/strock_semantics/semantics_hw/semantics_train",
#                        help='Training data directory')
#    parser.add_argument('--val_data_dir', type=str,
#                        default="/media/text/afedbba0-93b4-490d-9d12-20e5b7b22aa7/yan/exp2/S-Dauble-Attention/dataset/strock_semantics/semantics_hw/semantics_val",
#                        help='Validation data directory')
#    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
#    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading threads')
#
#    # Training parameters
#    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
#    parser.add_argument('--initial_lr', type=float, default=0.00025, help='Initial learning rate')
#    parser.add_argument('--num_stroke_classes', type=int, default=8, help='Number of stroke classes')
#
#    # Loss function parameters
#    parser.add_argument('--class_weights', type=str,
#                        default="1.35,1.25,1.25,1.0,1.0,1.25,1.25,1.35",
#                        help='Class weights (comma-separated)')
#
#    # Model parameters
#    parser.add_argument('--model_save_dir', type=str,
#                        default="/media/text/afedbba0-93b4-490d-9d12-20e5b7b22aa7/yan/exp2/S-Dauble-Attention/dataset/weight/hw_a",
#                        help='Model save directory')
#    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
#    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
#
#    # GPU settings
#    parser.add_argument('--gpu_ids', type=str, default="0,1,2,3", help='GPU IDs to use, e.g., "0,1,2,3"')
#
#    # Resume training
#    parser.add_argument('--continue_training', action='store_true', default=False,
#                        help='Resume training from checkpoint')
#    parser.add_argument('--pretrained_model_path', type=str, default="", help='Pretrained model path')
#    parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch')
#    parser.add_argument('--num_epochs_continue', type=int, default=100, help='Number of epochs to resume')
#
#    # Loss function type
#    parser.add_argument('--loss_type', type=str, default='weighted_bce', choices=['weighted_bce'],
#                        help='Loss function type')
#
#    # Visualisation parameters
#    parser.add_argument('--visualize_frequency', type=int, default=10, help='Visualisation save frequency (epochs)')
#    parser.add_argument('--target_size', type=int, default=1024, help='Visualisation image target size')
#    parser.add_argument('--border_width', type=int, default=64, help='Border width')
#    parser.add_argument('--vis_filtered_samples', action='store_true', default=True,
#                        help='Visualize samples filtered due to empty non-cross regions')
#
#    return parser
#
#
#class FastStrokeDataset(Dataset):
#    def __init__(self, data_dir, transform=None, num_workers=8, preload=False):
#        self.data_dir = data_dir
#        self.transform = transform
#        self.num_workers = num_workers
#        self.preload = preload
#
#        if not os.path.exists(data_dir):
#            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
#
#        self.image_folders = [f for f in glob.glob(os.path.join(data_dir, "image_*")) if os.path.isdir(f)]
#        self.image_folders.sort()
#        print(f"Found {len(self.image_folders)} image folders in {data_dir}")
#
#        self.data = []
#        for i, folder in enumerate(self.image_folders):
#            folder_name = os.path.basename(folder)
#            try:
#                original_files = glob.glob(os.path.join(folder, "*union*"))
#                cross_files = glob.glob(os.path.join(folder, "*intersection*"))
#                non_cross_files = glob.glob(os.path.join(folder, "*processed*"))
#                label_files = glob.glob(os.path.join(folder, "*label.npy"))
#
#                if original_files and cross_files and non_cross_files and label_files:
#                    self.data.append({
#                        "original": original_files[0],
#                        "cross": cross_files[0],
#                        "non_cross": non_cross_files[0],
#                        "label": label_files[0],
#                        "folder_name": folder_name
#                    })
#                    if i < 5 or i % 1000 == 0:
#                        print(f"Loading data, {i}/{len(self.image_folders)}: {folder_name}")
#                else:
#                    missing = []
#                    if not original_files:
#                        missing.append("union.png (original image)")
#                    if not cross_files:
#                        missing.append("_intersection.png (cross region image)")
#                    if not non_cross_files:
#                        missing.append("_final_processed.png (non-cross region image)")
#                    if not label_files:
#                        missing.append("label.npy (label)")
#                    print(f"Warning: Folder {folder_name} missing files: {missing}, skipped")
#            except Exception as e:
#                print(f"Error processing folder {folder_name}: {e}")
#
#        print(f"Total samples in {data_dir} dataset: {len(self.data)}")
#
#        if preload:
#            self._preload_data()
#
#    def _preload_data(self):
#        print(f"Preloading data with {self.num_workers} threads...")
#        start_time = time.time()
#        try:
#            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
#                self.preloaded_data = list(executor.map(lambda x: self.data[x], range(len(self.data))))
#        except Exception as e:
#            print(f"Data preloading error: {e}")
#            traceback.print_exc()
#            raise
#        end_time = time.time()
#        print(f"Data preloading completed, {len(self.preloaded_data)} samples, time taken: {end_time - start_time:.2f}s")
#
#    def __len__(self):
#        return len(self.data) if not self.preload else len(self.preloaded_data)
#
#    def __getitem__(self, idx):
#        if self.preload:
#            return self.preloaded_data[idx]
#        return self.data[idx]
#
#
#class SdaNetTrainer:
#    def __init__(self, args):
#        self.args = args
#        self.best_val_iou = -float('inf')
#        self.best_model_path = os.path.join(args.model_save_dir, "best_model.pth")
#        self.best_vis_dir = os.path.join(args.model_save_dir, "best_visualization")
#        self.best_info_path = os.path.join(args.model_save_dir, "best_info.txt")
#        self.start_time = time.time()
#
#        self.filtered_vis_dir = os.path.join(args.model_save_dir,
#                                             f"filtered_samples_vis_{time.strftime('%Y%m%d_%H%M%S')}")
#        if args.vis_filtered_samples:
#            os.makedirs(self.filtered_vis_dir, exist_ok=True)
#            print(f"Filtered samples (empty non-cross regions) will be saved to: {self.filtered_vis_dir}")
#
#        self.class_weights = torch.tensor([float(w) for w in args.class_weights.split(',')], dtype=torch.float32)
#        assert len(self.class_weights) == args.num_stroke_classes
#        print(f"Class weights: {self.class_weights.numpy()}")
#
#        self.target_size = (args.target_size, args.target_size)
#        self.border_color = (230, 230, 10)
#        self.border_width = args.border_width
#        np.random.seed(42)
#        self.colors = np.random.randint(50, 256, size=(args.num_stroke_classes, 3), dtype=np.uint8)
#        print(f"Generated {args.num_stroke_classes} colors, target size: {self.target_size[0]}x{self.target_size[1]}")
#
#        print("Trainer initialization completed")
#
#        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#        self.class_weights = self.class_weights.to(self.device)
#        print(f"Using device: {self.device}")
#
#        self.num_gpus = torch.cuda.device_count()
#        if self.num_gpus > 1:
#            print(f"Detected {self.num_gpus} GPUs, using DataParallel for multi-GPU training")
#        else:
#            print(f"Detected {self.num_gpus} GPU, using single GPU training")
#
#        os.makedirs(args.model_save_dir, exist_ok=True)
#        print(f"Model save directory: {args.model_save_dir}")
#
#        print(f"Initializing SdaNet")
#        self.model = SdaNet(
#            embed_dim=args.embed_dim,
#            num_heads=args.num_heads,
#            num_stroke_classes=args.num_stroke_classes,
#        ).to(self.device)
#        print("Model initialization completed")
#
#        try:
#            if torch.cuda.is_available():
#                torch.cuda.empty_cache()
#                for i in range(self.num_gpus):
#                    mem_info = torch.cuda.mem_get_info(i)
#                    print(f"GPU {i} memory: Free {mem_info[0] / 1024 ** 3:.2f} GB, Total {mem_info[1] / 1024 ** 3:.2f} GB")
#        except Exception as e:
#            print(f"Memory check error: {e}")
#            raise
#
#        if self.num_gpus > 1:
#            try:
#                self.model = nn.DataParallel(self.model)
#                print(f"DataParallel setup successful, using {self.num_gpus} GPUs")
#            except Exception as e:
#                print(f"DataParallel setup failed: {e}")
#                raise
#
#        self.optimizer = optim.Adam(self.model.parameters(), lr=args.initial_lr,
#                                    betas=(0.9, 0.999), weight_decay=1e-5)
#        print("Optimizer setup completed (Adam)")
#
#        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=60, eta_min=1e-6)
#        print(f"Learning rate scheduler setup completed, time taken: {time.time() - self.start_time:.2f}s")
#
#    def _get_real_model(self):
#        if isinstance(self.model, torch.nn.DataParallel):
#            return self.model.module
#        return self.model
#
#    def calculate_precision_recall_f1(self, pred_logits, label, eps=1e-8):
#        pred = (torch.sigmoid(pred_logits) > 0.5).float()
#        target = label.float()
#
#        tp = (pred * target).sum()
#        fp = (pred * (1 - target)).sum()
#        fn = ((1 - pred) * target).sum()
#
#        precision = tp / (tp + fp + eps)
#        recall = tp / (tp + fn + eps)
#        f1 = 2 * precision * recall / (precision + recall + eps)
#
#        return precision.item(), recall.item(), f1.item()
#
#    # ---------- Three metric IoU calculation ----------
#    def calculate_three_miou(self, pred_logits, label, eps=1e-6):
#        pred = (torch.sigmoid(pred_logits) > 0.5).float()
#        B, C, H, W = pred.shape
#    
#        total_list, cross_list, noncross_list = [], [], []
#    
#        for b in range(B):
#            p = pred[b]
#            t = label[b]
#            sum_t = t.sum(dim=0)
#    
#            cross_mask = (sum_t >= 2).float()
#            noncross_mask = (sum_t <= 1).float()
#    
#            # Total mIoU
#            t_ious = []
#            for c in range(C):
#                inter = (p[c] * t[c]).sum()
#                union = (p[c] + t[c]).sum() - inter
#               
#                iou = ((inter + eps) / (union + eps)).item()
#                t_ious.append(iou)
#            total_list.append(sum(t_ious) / C)
#    
#            # Cross mIoU
#            c_ious = []
#            for c in range(C):
#                inter = (p[c] * t[c] * cross_mask).sum()
#                union = ((p[c] + t[c]) * cross_mask).sum() - inter
#                if union > 0:
#                    iou = ((inter + eps) / (union + eps)).item()
#                    c_ious.append(iou)
#            cross_list.append(sum(c_ious) / len(c_ious) if c_ious else 1.0)
#    
#            # Non-Cross mIoU
#            n_ious = []
#            for c in range(C):
#                inter = (p[c] * t[c] * noncross_mask).sum()
#                union = ((p[c] + t[c]) * noncross_mask).sum() - inter
#                if union > 0:
#                    iou = ((inter + eps) / (union + eps)).item()
#                    n_ious.append(iou)
#            noncross_list.append(sum(n_ious) / len(n_ious) if n_ious else 1.0)
#
#   
#        return np.mean(total_list), np.mean(cross_list), np.mean(noncross_list)
#
#    # ---------- Loss functions with spatial weighting for non-cross pixels ----------
#    def weighted_bce_loss(self, pred_logits, target):
#        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
#        class_weights = self.class_weights.view(1, -1, 1, 1)
#
#        # Spatial weight: give higher weight to non-cross pixels (sum of classes <= 1)
#        sum_target = target.sum(dim=1, keepdim=True)          # [B, 1, H, W]
#        spatial_weight = torch.where(sum_target <= 1,
#                                     torch.tensor(1.5, device=target.device),
#                                     torch.tensor(1.0, device=target.device))
#
#        weighted_bce = bce * class_weights * spatial_weight
#        return weighted_bce.mean()
#
#    def combined_loss(self, pred_logits, target):
#        bce = self.weighted_bce_loss(pred_logits, target)
#        pred = torch.sigmoid(pred_logits)
#
#        inter = (pred * target).sum(dim=(2, 3))
#        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
#        dice = 1 - (2 * inter + 1e-5) / (union + 1e-5)
#        dice = dice.mean()
#
#        total_loss = 0.5 * bce + 0.5 * dice
#        return total_loss, bce, dice
#
#    # ---------- Sample processing with non-cross enhancement (training only) ----------
#    def process_sample(self, sample, is_train=True):
#        original_img = cv2.imread(sample["original"], cv2.IMREAD_GRAYSCALE)
#        cross_img = cv2.imread(sample["cross"], cv2.IMREAD_GRAYSCALE)
#        non_cross_img = cv2.imread(sample["non_cross"], cv2.IMREAD_GRAYSCALE)
#        label = np.load(sample["label"])
#        folder_name = sample["folder_name"]
#
#        if original_img is None:
#            raise ValueError(f"Failed to load original image, folder: {folder_name}, path: {sample['original']}")
#        if cross_img is None:
#            raise ValueError(f"Failed to cross region image, folder: {folder_name}, path: {sample['cross']}")
#        if non_cross_img is None:
#            raise ValueError(f"Failed to non-cross region image, folder: {folder_name}, path: {sample['non_cross']}")
#
#        if original_img.shape != (128, 128):
#            original_img = cv2.resize(original_img, (128, 128), interpolation=cv2.INTER_NEAREST)
#        if cross_img.shape != (128, 128):
#            cross_img = cv2.resize(cross_img, (128, 128), interpolation=cv2.INTER_NEAREST)
#        if non_cross_img.shape != (128, 128):
#            non_cross_img = cv2.resize(non_cross_img, (128, 128), interpolation=cv2.INTER_NEAREST)
#
#        if label.shape[0] != 128 or label.shape[1] != 128:
#            label_resized = np.zeros((128, 128, label.shape[2]), dtype=label.dtype)
#            for c in range(label.shape[2]):
#                label_resized[:, :, c] = cv2.resize(label[:, :, c], (128, 128), interpolation=cv2.INTER_NEAREST)
#            label = label_resized
#
#        original_tensor = torch.from_numpy(original_img).float().unsqueeze(0) / 255.0
#        cross_tensor = torch.from_numpy(cross_img).float().unsqueeze(0) / 255.0
#        non_cross_tensor = torch.from_numpy(non_cross_img).float().unsqueeze(0) / 255.0
#
#        if is_train:
#            # Reduced enhancement factor + small Gaussian noise for generalisation
#            non_cross_tensor = non_cross_tensor * 1.15
#            non_cross_tensor = non_cross_tensor + torch.randn_like(non_cross_tensor) * 0.01
#            non_cross_tensor = torch.clamp(non_cross_tensor, 0, 1)
#
#        label_tensor = torch.from_numpy(label).float().permute(2, 0, 1)
#        return original_tensor, cross_tensor, non_cross_tensor, label_tensor, folder_name
#
#    # ---------- Visualisation functions (kept from original) ----------
#    def get_color_image(self, single_image, white_bg=True, apply_threshold=True):
#        if isinstance(single_image, torch.Tensor):
#            single_image = single_image.detach().cpu().numpy()
#        if apply_threshold and len(single_image.shape) == 3:
#            single_image = (torch.sigmoid(torch.from_numpy(single_image)) > 0.5).numpy()
#        C, H, W = single_image.shape
#        color_img = np.full((H, W, 3), 255, dtype=np.uint8) if white_bg else np.zeros((H, W, 3), dtype=np.uint8)
#        for c in range(C):
#            if c >= len(self.colors):
#                continue
#            mask = single_image[c] > 0.5
#            color = self.colors[c]
#            color_img[mask] = np.clip(color_img[mask] + color, 0, 255)
#        return color_img
#
#    def add_border(self, img, color, width):
#        h, w = img.shape[:2]
#        return cv2.rectangle(img.copy(), (0, 0), (w - 1, h - 1), color, thickness=width)
#
#    def visualize_epoch_predictions(self, epoch):
#        epoch_vis_dir = os.path.join(self.args.model_save_dir, f"epoch_{epoch}_visualization")
#        if os.path.exists(epoch_vis_dir):
#            shutil.rmtree(epoch_vis_dir)
#        os.makedirs(epoch_vis_dir, exist_ok=True)
#
#        test_dataset = self.val_loader.dataset
#        total_samples = len(test_dataset)
#        print(f"\n[Visualisation] Starting epoch {epoch} visualisation, processing {total_samples} samples, save path: {epoch_vis_dir}")
#
#        self.model.eval()
#        with torch.no_grad():
#            for vis_idx, data_idx in enumerate(range(total_samples)):
#                try:
#                    sample = test_dataset[data_idx]
#                    result = self.process_sample(sample, is_train=False)  # No enhancement for visualisation
#                    if result is None:
#                        continue
#                    original_img, cross_img, non_cross_img, label, folder_name = result
#
#                    original_batch = original_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    cross_batch = cross_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    non_cross_batch = non_cross_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    pred_logits, _ = self.model(cross_batch, non_cross_batch, original_batch)
#                    pred_prob = torch.sigmoid(pred_logits)
#
#                    original_np = original_img.squeeze().cpu().numpy()
#                    original_rgb = cv2.cvtColor((original_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    original_rgb_target = cv2.resize(original_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    cross_np = cross_img.squeeze().cpu().numpy()
#                    cross_rgb = cv2.cvtColor((cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    cross_rgb_target = cv2.resize(cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    non_cross_np = non_cross_img.squeeze().cpu().numpy()
#                    non_cross_rgb = cv2.cvtColor((non_cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    non_cross_rgb_target = cv2.resize(non_cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    label_np = label.cpu().numpy()
#                    label_color = self.get_color_image(label_np, white_bg=True, apply_threshold=False)
#                    label_color_target = cv2.resize(label_color, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    pred_np = pred_prob.squeeze(0).cpu().numpy()
#                    pred_color = self.get_color_image(pred_np, white_bg=True, apply_threshold=True)
#                    pred_color_target = cv2.resize(pred_color, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    original_with_border = self.add_border(original_rgb_target, self.border_color, self.border_width)
#                    cross_with_border = self.add_border(cross_rgb_target, self.border_color, self.border_width)
#                    non_cross_with_border = self.add_border(non_cross_rgb_target, self.border_color, self.border_width)
#                    label_with_border = self.add_border(label_color_target, self.border_color, self.border_width)
#                    prediction_with_border = self.add_border(pred_color_target, self.border_color, self.border_width)
#
#                    save_paths = {
#                        "original": os.path.join(epoch_vis_dir, f"{folder_name}_original_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "cross": os.path.join(epoch_vis_dir, f"{folder_name}_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "non_cross": os.path.join(epoch_vis_dir, f"{folder_name}_non_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "label": os.path.join(epoch_vis_dir, f"{folder_name}_label_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "prediction": os.path.join(epoch_vis_dir, f"{folder_name}_prediction_{self.target_size[0]}x{self.target_size[1]}.png")
#                    }
#                    for name, path in save_paths.items():
#                        cv2.imwrite(path, locals()[name + "_with_border"])
#
#                    if (vis_idx + 1) % 10 == 0 or vis_idx == total_samples - 1:
#                        print(f"[Visualisation] Epoch {epoch} progress: {vis_idx + 1}/{total_samples} samples, current folder: {folder_name}")
#                except Exception as e:
#                    print(f"[Visualisation Error] Epoch {epoch} sample {folder_name} failed: {str(e)}")
#                    continue
#        print(f"[Visualisation] Epoch {epoch} visualisation completed, saved to: {epoch_vis_dir}\n")
#
#    def save_visualizations(self):
#        if os.path.exists(self.best_vis_dir):
#            shutil.rmtree(self.best_vis_dir)
#        os.makedirs(self.best_vis_dir, exist_ok=True)
#
#        try:
#            model_to_load = self._get_real_model()
#            model_to_load.load_state_dict(torch.load(self.best_model_path))
#            print(f"Loaded best model weights for visualisation: {self.best_model_path}")
#        except Exception as e:
#            print(f"Failed to load best model weights, using current model state for visualisation: {e}")
#
#        test_dataset = self.val_loader.dataset
#        total_samples = len(test_dataset)
#        print(f"\n[Best Model Visualisation] Generating visualisations for {total_samples} validation samples, save path: {self.best_vis_dir}")
#
#        self.model.eval()
#        with torch.no_grad():
#            for vis_idx, data_idx in enumerate(range(total_samples)):
#                try:
#                    sample = test_dataset[data_idx]
#                    result = self.process_sample(sample, is_train=False)
#                    if result is None:
#                        continue
#                    original_img, cross_img, non_cross_img, label, folder_name = result
#
#                    original_batch = original_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    cross_batch = cross_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    non_cross_batch = non_cross_img.unsqueeze(0).to(self.device, non_blocking=True)
#                    pred_logits, _ = self.model(cross_batch, non_cross_batch, original_batch)
#                    pred_prob = torch.sigmoid(pred_logits)
#
#                    original_np = original_img.squeeze().cpu().numpy()
#                    original_rgb = cv2.cvtColor((original_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    original_rgb_target = cv2.resize(original_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    cross_np = cross_img.squeeze().cpu().numpy()
#                    cross_rgb = cv2.cvtColor((cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    cross_rgb_target = cv2.resize(cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    non_cross_np = non_cross_img.squeeze().cpu().numpy()
#                    non_cross_rgb = cv2.cvtColor((non_cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
#                    non_cross_rgb_target = cv2.resize(non_cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    label_np = label.cpu().numpy()
#                    label_color = self.get_color_image(label_np, white_bg=True, apply_threshold=False)
#                    label_color_target = cv2.resize(label_color, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    pred_np = pred_prob.squeeze(0).cpu().numpy()
#                    pred_color = self.get_color_image(pred_np, white_bg=True, apply_threshold=False)
#                    pred_color_target = cv2.resize(pred_color, self.target_size, interpolation=cv2.INTER_LINEAR)
#
#                    original_with_border = self.add_border(original_rgb_target, self.border_color, self.border_width)
#                    cross_with_border = self.add_border(cross_rgb_target, self.border_color, self.border_width)
#                    non_cross_with_border = self.add_border(non_cross_rgb_target, self.border_color, self.border_width)
#                    label_with_border = self.add_border(label_color_target, self.border_color, self.border_width)
#                    prediction_with_border = self.add_border(pred_color_target, self.border_color, self.border_width)
#
#                    save_paths = {
#                        "original": os.path.join(self.best_vis_dir, f"{folder_name}_original_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "cross": os.path.join(self.best_vis_dir, f"{folder_name}_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "non_cross": os.path.join(self.best_vis_dir, f"{folder_name}_non_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "label": os.path.join(self.best_vis_dir, f"{folder_name}_label_{self.target_size[0]}x{self.target_size[1]}.png"),
#                        "prediction": os.path.join(self.best_vis_dir, f"{folder_name}_prediction_{self.target_size[0]}x{self.target_size[1]}.png")
#                    }
#                    for name, path in save_paths.items():
#                        cv2.imwrite(path, locals()[name + "_with_border"])
#
#                    if (vis_idx + 1) % 10 == 0 or vis_idx == total_samples - 1:
#                        print(f"[Best Model Visualisation] Progress: {vis_idx + 1}/{total_samples} samples, current folder: {folder_name}")
#                except Exception as e:
#                    print(f"[Best Model Visualisation Error] Sample {folder_name} failed: {str(e)}")
#                    continue
#        print(f"[Best Model Visualisation] All visualisations completed, saved to: {self.best_vis_dir}\n")
#
#    def calculate_final_metrics(self):
#        self.model.eval()
#        all_p, all_r, all_f1 = [], [], []
#        with torch.no_grad():
#            for batch in self.val_loader:
#                orig_list, cross_list, noncross_list, label_list = [], [], [], []
#                for i in range(len(batch["original"])):
#                    sample = {
#                        "original": batch["original"][i],
#                        "cross": batch["cross"][i],
#                        "non_cross": batch["non_cross"][i],
#                        "label": batch["label"][i],
#                        "folder_name": batch["folder_name"][i]
#                    }
#                    res = self.process_sample(sample, is_train=False)
#                    if res:
#                        o, c, n, l, _ = res
#                        orig_list.append(o)
#                        cross_list.append(c)
#                        noncross_list.append(n)
#                        label_list.append(l)
#                if not orig_list:
#                    continue
#                orig_batch = torch.stack(orig_list).to(self.device)
#                cross_batch = torch.stack(cross_list).to(self.device)
#                noncross_batch = torch.stack(noncross_list).to(self.device)
#                label_batch = torch.stack(label_list).to(self.device)
#                pred_logits, _ = self.model(cross_batch, noncross_batch, orig_batch)
#                p, r, f1 = self.calculate_precision_recall_f1(pred_logits, label_batch)
#                all_p.append(p)
#                all_r.append(r)
#                all_f1.append(f1)
#        return np.mean(all_p), np.mean(all_r), np.mean(all_f1)
#
#    # ---------- Data preparation ----------
#    def prepare_data(self):
#        print("Preparing data loaders")
#        start_time = time.time()
#        transform = transforms.Compose([transforms.ToTensor()])
#
#        train_dataset = FastStrokeDataset(self.args.data_dir, transform=transform,
#                                          num_workers=self.args.num_workers, preload=True)
#        val_dataset = FastStrokeDataset(self.args.val_data_dir, transform=transform,
#                                        num_workers=self.args.num_workers, preload=True)
#
#        train_size = len(train_dataset)
#        val_size = len(val_dataset)
#        print(f"Data preparation completed, training samples: {train_size}, validation samples: {val_size}, time taken: {time.time() - start_time:.2f}s")
#
#        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True,
#                                       num_workers=self.args.num_workers, pin_memory=True)
#        self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False,
#                                     num_workers=self.args.num_workers // 2, pin_memory=True)
#        print(f"Training data loader setup completed, batch size: {self.args.batch_size}, threads: {self.args.num_workers}")
#        print(f"Validation data loader setup completed, batch size: {self.args.batch_size}, threads: {self.args.num_workers // 2}")
#        print(f"Total data preparation time: {time.time() - start_time:.2f}s")
#        return self.train_loader, self.val_loader
#
#    # ---------- Main training loop with three metrics ----------
#    def train(self):
#        print("Starting SdaNet training with three metrics and spatial weighting")
#        start_epoch = 1 if not self.args.continue_training else self.args.start_epoch
#        total_epochs = self.args.epochs if not self.args.continue_training else self.args.start_epoch + self.args.num_epochs_continue - 1
#        print(f"Starting epoch: {start_epoch}, Total training epochs: {total_epochs}")
#
#        for epoch in range(start_epoch, total_epochs + 1):
#            epoch_start = time.time()
#            current_lr = self.optimizer.param_groups[0]['lr']
#            print(f"\nEpoch {epoch}/{total_epochs} started, LR={current_lr:.8f}")
#            print("=" * 60)
#
#            # ---------- Training phase ----------
#            self.model.train()
#            train_total_loss = 0.0
#            train_bce_loss = 0.0
#            train_total_miou, train_cross_miou, train_nc_miou = 0.0, 0.0, 0.0
#            train_p, train_r, train_f1 = 0.0, 0.0, 0.0
#            batch_count = len(self.train_loader)
#
#            for batch_idx, batch in enumerate(self.train_loader):
#                orig_list, cross_list, noncross_list, label_list = [], [], [], []
#                for i in range(len(batch["original"])):
#                    sample = {
#                        "original": batch["original"][i],
#                        "cross": batch["cross"][i],
#                        "non_cross": batch["non_cross"][i],
#                        "label": batch["label"][i],
#                        "folder_name": batch["folder_name"][i]
#                    }
#                    result = self.process_sample(sample, is_train=True)
#                    if result is None:
#                        continue
#                    o, c, n, l, _ = result
#                    orig_list.append(o)
#                    cross_list.append(c)
#                    noncross_list.append(n)
#                    label_list.append(l)
#
#                if len(orig_list) == 0:
#                    continue
#
#                orig_batch = torch.stack(orig_list).to(self.device)
#                cross_batch = torch.stack(cross_list).to(self.device)
#                noncross_batch = torch.stack(noncross_list).to(self.device)
#                label_batch = torch.stack(label_list).to(self.device)
#
#                self.optimizer.zero_grad()
#                pred_logits, _ = self.model(cross_batch, noncross_batch, orig_batch)
#                total_loss, bce_loss, _ = self.combined_loss(pred_logits, label_batch)
#                total_miou, cross_miou, nc_miou = self.calculate_three_miou(pred_logits, label_batch)
#                p, r, f1 = self.calculate_precision_recall_f1(pred_logits, label_batch)
#
#                total_loss.backward()
#                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
#                self.optimizer.step()
#
#                train_total_loss += total_loss.item()
#                train_bce_loss += bce_loss.item()
#                train_total_miou += total_miou
#                train_cross_miou += cross_miou
#                train_nc_miou += nc_miou
#                train_p += p
#                train_r += r
#                train_f1 += f1
#
#                if batch_idx % 10 == 0 or batch_idx == batch_count - 1:
#                    print(f"[Train {batch_idx+1}/{batch_count}] Loss:{total_loss.item():.3f} | Total:{total_miou:.3f} | Cross:{cross_miou:.3f} | NonCross:{nc_miou:.3f} | P:{p:.3f} R:{r:.3f} F1:{f1:.3f}")
#
#            train_total_loss /= batch_count
#            train_bce_loss /= batch_count
#            train_total_miou /= batch_count
#            train_cross_miou /= batch_count
#            train_nc_miou /= batch_count
#            train_p /= batch_count
#            train_r /= batch_count
#            train_f1 /= batch_count
#
#            print(f"[Train Done] Loss:{train_total_loss:.3f} | Total:{train_total_miou:.3f} Cross:{train_cross_miou:.3f} NonCross:{train_nc_miou:.3f} | P:{train_p:.3f} R:{train_r:.3f} F1:{train_f1:.3f}")
#
#            # ---------- Validation phase ----------
#            self.model.eval()
#            val_total_loss = 0.0
#            val_bce_loss = 0.0
#            val_total_miou, val_cross_miou, val_nc_miou = 0.0, 0.0, 0.0
#            val_p, val_r, val_f1 = 0.0, 0.0, 0.0
#            val_batch_count = len(self.val_loader)
#
#            with torch.no_grad():
#                for batch_idx, batch in enumerate(self.val_loader):
#                    orig_list, cross_list, noncross_list, label_list = [], [], [], []
#                    for i in range(len(batch["original"])):
#                        sample = {
#                            "original": batch["original"][i],
#                            "cross": batch["cross"][i],
#                            "non_cross": batch["non_cross"][i],
#                            "label": batch["label"][i],
#                            "folder_name": batch["folder_name"][i]
#                        }
#                        result = self.process_sample(sample, is_train=False)
#                        if result is None:
#                            continue
#                        o, c, n, l, _ = result
#                        orig_list.append(o)
#                        cross_list.append(c)
#                        noncross_list.append(n)
#                        label_list.append(l)
#
#                    if len(orig_list) == 0:
#                        continue
#
#                    orig_batch = torch.stack(orig_list).to(self.device)
#                    cross_batch = torch.stack(cross_list).to(self.device)
#                    noncross_batch = torch.stack(noncross_list).to(self.device)
#                    label_batch = torch.stack(label_list).to(self.device)
#
#                    pred_logits, _ = self.model(cross_batch, noncross_batch, orig_batch)
#                    total_loss, bce_loss, _ = self.combined_loss(pred_logits, label_batch)
#                    total_miou, cross_miou, nc_miou = self.calculate_three_miou(pred_logits, label_batch)
#                    p, r, f1 = self.calculate_precision_recall_f1(pred_logits, label_batch)
#
#                    val_total_loss += total_loss.item()
#                    val_bce_loss += bce_loss.item()
#                    val_total_miou += total_miou
#                    val_cross_miou += cross_miou
#                    val_nc_miou += nc_miou
#                    val_p += p
#                    val_r += r
#                    val_f1 += f1
#
#                    if batch_idx % 10 == 0 or batch_idx == val_batch_count - 1:
#                        print(f"[Val {batch_idx+1}/{val_batch_count}] Loss:{total_loss.item():.3f} | Total:{total_miou:.3f} Cross:{cross_miou:.3f} NonCross:{nc_miou:.3f} | P:{p:.3f} R:{r:.3f} F1:{f1:.3f}")
#
#            val_total_loss /= val_batch_count
#            val_bce_loss /= val_batch_count
#            val_total_miou /= val_batch_count
#            val_cross_miou /= val_batch_count
#            val_nc_miou /= val_batch_count
#            val_p /= val_batch_count
#            val_r /= val_batch_count
#            val_f1 /= val_batch_count
#
#            print(f"[Val Done] Loss:{val_total_loss:.3f} | Total:{val_total_miou:.3f} Cross:{val_cross_miou:.3f} NonCross:{val_nc_miou:.3f} | P:{val_p:.3f} R:{val_r:.3f} F1:{val_f1:.3f}")
#
#            self.scheduler.step()
#
#            # Save best model based on Total mIoU
#            if val_total_miou > self.best_val_iou:
#                self.best_val_iou = val_total_miou
#                self.save_model(epoch, val_total_loss, val_total_miou)
#                self.save_best_info_with_metrics(val_total_miou, val_cross_miou, val_nc_miou, val_p, val_r, val_f1)
#
#            # Visualisation
#            if epoch % self.args.visualize_frequency == 0:
#                self.visualize_epoch_predictions(epoch)
#
#            epoch_time = time.time() - epoch_start
#            print(f"Epoch {epoch} finished in {epoch_time:.2f}s\n")
#
#        # Post-training
#        print("\n" + "="*60)
#        final_p, final_r, final_f1 = self.calculate_final_metrics()
#        print(f"[Final Metrics] Precision: {final_p:.4f} | Recall: {final_r:.4f} | F1: {final_f1:.4f}")
#        print("Training completed, generating best model visualisations...")
#        self.save_visualizations()
#        print(f"Total training time: {time.time() - self.start_time:.2f}s")
#        print(f"Best Total mIoU: {self.best_val_iou:.6f}")
#        print("="*60)
#
#    def save_model(self, epoch, val_loss, val_iou):
#        try:
#            model_to_save = self._get_real_model()
#            torch.save(model_to_save.state_dict(), self.best_model_path)
#            print(f"Best model saved: {self.best_model_path} (Epoch: {epoch}, Total mIoU: {val_iou:.6f})")
#        except Exception as e:
#            print(f"Model save failed: {e}")
#            raise
#
#    def save_best_info_with_metrics(self, total_miou, cross_miou, noncross_miou, precision, recall, f1):
#        with open(self.best_info_path, 'w') as f:
#            f.write("Best model metrics for paper:\n")
#            f.write(f"Total mIoU: {total_miou:.6f}\n")
#            f.write(f"Cross mIoU: {cross_miou:.6f}\n")
#            f.write(f"Non-Cross mIoU: {noncross_miou:.6f}\n")
#            f.write(f"Precision: {precision:.6f}\n")
#            f.write(f"Recall: {recall:.6f}\n")
#            f.write(f"F1-Score: {f1:.6f}\n")
#            f.write(f"Class weights: {self.class_weights.cpu().numpy()}\n")
#            f.write(f"LR: {self.optimizer.param_groups[0]['lr']:.8f}\n")
#            f.write(f"Embed dim: {self.args.embed_dim}\n")
#            f.write(f"Heads: {self.args.num_heads}\n")
#            f.write(f"Classes: {self.args.num_stroke_classes}\n")
#            f.write(f"Batch size: {self.args.batch_size}\n")
#        print(f"Best info saved to: {self.best_info_path}")
#
#
#def main():
#    parser = get_parser()
#    args = parser.parse_args()
#
#    if not os.path.exists(args.val_data_dir):
#        raise ValueError(f"Validation data directory does not exist: {args.val_data_dir}")
#    if args.continue_training and args.pretrained_model_path == "":
#        print("Error: --continue_training specified but --pretrained_model_path not provided")
#        sys.exit(1)
#    if len(args.class_weights.split(',')) != args.num_stroke_classes:
#        raise ValueError(f"Number of class weights must match number of classes ({args.num_stroke_classes})")
#
#    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
#    if torch.cuda.is_available():
#        torch.cuda.empty_cache()
#        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
#
#    print("="*60)
#    print("SdaNet Training Configuration (Three Metrics + Spatial Weighting)")
#    print(f"Python version: {sys.version.split()[0]}")
#    print(f"PyTorch version: {torch.__version__}")
#    print(f"CUDA version: {torch.version.cuda}")
#    print(f"Using GPU IDs: {args.gpu_ids}")
#    print(f"Model save directory: {args.model_save_dir}")
#    print(f"Class weights: {args.class_weights}")
#    print(f"Visualize filtered samples: {args.vis_filtered_samples}")
#    print(f"Save validation sample visualisation every {args.visualize_frequency} epochs")
#    print("="*60)
#
#    try:
#        trainer = SdaNetTrainer(args)
#        trainer.prepare_data()
#        trainer.train()
#    except Exception as e:
#        print(f"Training process failed: {str(e)}")
#        traceback.print_exc()
#        sys.exit(1)
#
#
#if __name__ == "__main__":
#    main()






## -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'  
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import cv2

import glob
import time
import sys
import traceback
import shutil
from torchvision import transforms
import argparse
from concurrent.futures import ThreadPoolExecutor
import torch.optim.lr_scheduler as lr_scheduler

from model.SdaNet import SdaNet

num_gpus = torch.cuda.device_count()
use_data_parallel = num_gpus >= 2
print(f"Detected {num_gpus} GPUs (Use DataParallel: {use_data_parallel})")


def get_parser():
    parser = argparse.ArgumentParser(description='SdaNet Training with Three Metrics and Spatial Weighting')

    # Data parameters
    parser.add_argument('--data_dir', type=str,
                        default="/media/text/afedbba0-93b4-490d-9d12-20e5b7b22aa7/yan/exp2/S-Dauble-Attention/dataset/strock_semantics/semantics_kai/semantics_train",
                        help='Training data directory')
    parser.add_argument('--val_data_dir', type=str,
                        default="/media/text/afedbba0-93b4-490d-9d12-20e5b7b22aa7/yan/exp2/S-Dauble-Attention/dataset/strock_semantics/semantics_kai/semantics_val",
                        help='Validation data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading threads')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Total training epochs')
    parser.add_argument('--initial_lr', type=float, default=0.00025, help='Initial learning rate')
    parser.add_argument('--num_stroke_classes', type=int, default=8, help='Number of stroke classes')

    # Loss function parameters
    parser.add_argument('--class_weights', type=str,
                        default="1.35,1.25,1.25,1.0,1.0,1.25,1.25,1.35",
                        help='Class weights (comma-separated)')

    # Model parameters
    parser.add_argument('--model_save_dir', type=str,
                        default="/media/text/afedbba0-93b4-490d-9d12-20e5b7b22aa7/yan/exp2/S-Dauble-Attention/dataset/weight/connection_kai",
                        help='Model save directory')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')

    # GPU settings
    parser.add_argument('--gpu_ids', type=str, default="2,3", help='GPU IDs to use, e.g., "2,3"')

    # Resume training
    parser.add_argument('--continue_training', action='store_true', default=False,
                        help='Resume training from checkpoint')
    parser.add_argument('--pretrained_model_path', type=str, default="", help='Pretrained model path')
    parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch')
    parser.add_argument('--num_epochs_continue', type=int, default=100, help='Number of epochs to resume')
    parser.add_argument('--no_skip_connections', action='store_true', default=False,
                    help='Disable skip connections in decoder (for ablation)')

    # Loss function type
    parser.add_argument('--loss_type', type=str, default='weighted_bce', choices=['weighted_bce'],
                        help='Loss function type')

    # Visualisation parameters
    parser.add_argument('--visualize_frequency', type=int, default=10, help='Visualisation save frequency (epochs)')
    parser.add_argument('--target_size', type=int, default=1024, help='Visualisation image target size')
    parser.add_argument('--border_width', type=int, default=64, help='Border width')
    parser.add_argument('--vis_filtered_samples', action='store_true', default=True,
                        help='Visualize samples filtered due to empty non-cross regions')

    return parser


class FastStrokeDataset(Dataset):
    def __init__(self, data_dir, transform=None, num_workers=8, preload=False):
        self.data_dir = data_dir
        self.transform = transform
        self.num_workers = num_workers
        self.preload = preload

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

        self.image_folders = [f for f in glob.glob(os.path.join(data_dir, "image_*")) if os.path.isdir(f)]
        self.image_folders.sort()
        print(f"Found {len(self.image_folders)} image folders in {data_dir}")

        self.data = []
        for i, folder in enumerate(self.image_folders):
            folder_name = os.path.basename(folder)
            try:
                original_files = glob.glob(os.path.join(folder, "*union*"))
                cross_files = glob.glob(os.path.join(folder, "*intersection*"))
                non_cross_files = glob.glob(os.path.join(folder, "*processed*"))
                label_files = glob.glob(os.path.join(folder, "*label.npy"))

                if original_files and cross_files and non_cross_files and label_files:
                    self.data.append({
                        "original": original_files[0],
                        "cross": cross_files[0],
                        "non_cross": non_cross_files[0],
                        "label": label_files[0],
                        "folder_name": folder_name
                    })
                    if i < 5 or i % 1000 == 0:
                        print(f"Loading data, {i}/{len(self.image_folders)}: {folder_name}")
                else:
                    missing = []
                    if not original_files:
                        missing.append("union.png (original image)")
                    if not cross_files:
                        missing.append("_intersection.png (cross region image)")
                    if not non_cross_files:
                        missing.append("_final_processed.png (non-cross region image)")
                    if not label_files:
                        missing.append("label.npy (label)")
                    print(f"Warning: Folder {folder_name} missing files: {missing}, skipped")
            except Exception as e:
                print(f"Error processing folder {folder_name}: {e}")

        print(f"Total samples in {data_dir} dataset: {len(self.data)}")

        if preload:
            self._preload_data()

    def _preload_data(self):
        print(f"Preloading data with {self.num_workers} threads...")
        start_time = time.time()
        try:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                self.preloaded_data = list(executor.map(lambda x: self.data[x], range(len(self.data))))
        except Exception as e:
            print(f"Data preloading error: {e}")
            traceback.print_exc()
            raise
        end_time = time.time()
        print(f"Data preloading completed, {len(self.preloaded_data)} samples, time taken: {end_time - start_time:.2f}s")

    def __len__(self):
        return len(self.data) if not self.preload else len(self.preloaded_data)

    def __getitem__(self, idx):
        if self.preload:
            return self.preloaded_data[idx]
        return self.data[idx]


class SdaNetTrainer:
    def __init__(self, args):
        self.args = args
        self.best_val_iou = -float('inf')
        self.best_model_path = os.path.join(args.model_save_dir, "best_model.pth")
        self.best_vis_dir = os.path.join(args.model_save_dir, "best_visualization")
        self.best_info_path = os.path.join(args.model_save_dir, "best_info.txt")
        self.start_time = time.time()

        self.filtered_vis_dir = os.path.join(args.model_save_dir,
                                             f"filtered_samples_vis_{time.strftime('%Y%m%d_%H%M%S')}")
        if args.vis_filtered_samples:
            os.makedirs(self.filtered_vis_dir, exist_ok=True)
            print(f"Filtered samples (empty non-cross regions) will be saved to: {self.filtered_vis_dir}")

        self.class_weights = torch.tensor([float(w) for w in args.class_weights.split(',')], dtype=torch.float32)
        assert len(self.class_weights) == args.num_stroke_classes
        print(f"Class weights: {self.class_weights.numpy()}")

        self.target_size = (args.target_size, args.target_size)
        self.border_color = (230, 230, 10)
        self.border_width = args.border_width
        np.random.seed(2024)
        self.colors = np.random.randint(50, 256, size=(args.num_stroke_classes, 3), dtype=np.uint8)
        print(f"Generated {args.num_stroke_classes} colors, target size: {self.target_size[0]}x{self.target_size[1]}")

        print("Trainer initialization completed")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_weights = self.class_weights.to(self.device)
        print(f"Using device: {self.device}")

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"Detected {self.num_gpus} GPUs, using DataParallel for multi-GPU training")
        else:
            print(f"Detected {self.num_gpus} GPU, using single GPU training")

        os.makedirs(args.model_save_dir, exist_ok=True)
        print(f"Model save directory: {args.model_save_dir}")
        
        use_skip = not args.no_skip_connections 
        print(f"Use skip connections: {use_skip}")
        
        print(f"Initializing SdaNet")
        self.model = SdaNet(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_stroke_classes=args.num_stroke_classes,
            encoder_mode='D', 
            use_skip_connections=use_skip,
        ).to(self.device)
        print("Model initialization completed")

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                for i in range(self.num_gpus):
                    mem_info = torch.cuda.mem_get_info(i)
                    print(f"GPU {i} memory: Free {mem_info[0] / 1024 ** 3:.2f} GB, Total {mem_info[1] / 1024 ** 3:.2f} GB")
        except Exception as e:
            print(f"Memory check error: {e}")
            raise

        if self.num_gpus > 1:
            try:
                self.model = nn.DataParallel(self.model)
                print(f"DataParallel setup successful, using {self.num_gpus} GPUs")
            except Exception as e:
                print(f"DataParallel setup failed: {e}")
                raise

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.initial_lr,
                                    betas=(0.9, 0.999), weight_decay=1e-5)
        print("Optimizer setup completed (Adam)")

        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=60, eta_min=1e-6)
        print(f"Learning rate scheduler setup completed, time taken: {time.time() - self.start_time:.2f}s")

    def _get_real_model(self):
        if isinstance(self.model, torch.nn.DataParallel):
            return self.model.module
        return self.model

    def calculate_precision_recall_f1(self, pred_logits, label, eps=1e-8):
        pred = (torch.sigmoid(pred_logits) > 0.5).float()
        target = label.float()

        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        return precision.item(), recall.item(), f1.item()

    # ---------- Three metric IoU calculation ----------
    def calculate_three_miou(self, pred_logits, label, eps=1e-6):
        pred = (torch.sigmoid(pred_logits) > 0.5).float()
        B, C, H, W = pred.shape
    
        total_list, cross_list, noncross_list = [], [], []
    
        for b in range(B):
            p = pred[b]          # [C, H, W]
            t = label[b]         # [C, H, W]
            sum_t = t.sum(dim=0) # [H, W]
    
            cross_mask = (sum_t >= 2).float()
            noncross_mask = (sum_t <= 1).float()
    
            # ---------- Total mIoU ----------
            t_ious = []
            for c in range(C):
                if t[c].sum() > 0:  
                    inter = (p[c] * t[c]).sum()
                    union = (p[c] + t[c]).sum() - inter
                    iou = (inter + eps) / (union + eps)
                    t_ious.append(iou.item())
            total_miou = sum(t_ious) / len(t_ious) if t_ious else 1.0
            total_list.append(total_miou)
    
            # ---------- Cross mIoU ----------
            c_ious = []
            for c in range(C):
                t_cross = t[c] * cross_mask
                if t_cross.sum() > 0:  
                    inter = (p[c] * t[c] * cross_mask).sum()
                    union = ((p[c] + t[c]) * cross_mask).sum() - inter
                    iou = (inter + eps) / (union + eps)
                    c_ious.append(iou.item())
            cross_miou = sum(c_ious) / len(c_ious) if c_ious else 1.0
            cross_list.append(cross_miou)
    
            # ---------- Non?Cross mIoU ----------
            n_ious = []
            for c in range(C):
                t_nc = t[c] * noncross_mask
                if t_nc.sum() > 0:   
                    inter = (p[c] * t[c] * noncross_mask).sum()
                    union = ((p[c] + t[c]) * noncross_mask).sum() - inter
                    iou = (inter + eps) / (union + eps)
                    n_ious.append(iou.item())
            noncross_miou = sum(n_ious) / len(n_ious) if n_ious else 1.0
            noncross_list.append(noncross_miou)
    
        return np.mean(total_list), np.mean(cross_list), np.mean(noncross_list)

    # ---------- Loss functions with spatial weighting for non-cross pixels ----------
    def weighted_bce_loss(self, pred_logits, target):
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        class_weights = self.class_weights.view(1, -1, 1, 1)

        # Spatial weight: give higher weight to non-cross pixels (sum of classes <= 1)
        sum_target = target.sum(dim=1, keepdim=True)          # [B, 1, H, W]
        spatial_weight = torch.where(sum_target <= 1,
                                     torch.tensor(1.5, device=target.device),
                                     torch.tensor(1.0, device=target.device))

        weighted_bce = bce * class_weights * spatial_weight
        return weighted_bce.mean()

    def combined_loss(self, pred_logits, target):
        bce = self.weighted_bce_loss(pred_logits, target)
        pred = torch.sigmoid(pred_logits)

        inter = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 1 - (2 * inter + 1e-5) / (union + 1e-5)
        dice = dice.mean()

        total_loss = 0.5 * bce + 0.5 * dice
        return total_loss, bce, dice

    # ---------- Sample processing with non-cross enhancement (training only) ----------
    def process_sample(self, sample, is_train=True):
        original_img = cv2.imread(sample["original"], cv2.IMREAD_GRAYSCALE)
        cross_img = cv2.imread(sample["cross"], cv2.IMREAD_GRAYSCALE)
        non_cross_img = cv2.imread(sample["non_cross"], cv2.IMREAD_GRAYSCALE)
        label = np.load(sample["label"])
        folder_name = sample["folder_name"]

        if original_img is None:
            raise ValueError(f"Failed to load original image, folder: {folder_name}, path: {sample['original']}")
        if cross_img is None:
            raise ValueError(f"Failed to cross region image, folder: {folder_name}, path: {sample['cross']}")
        if non_cross_img is None:
            raise ValueError(f"Failed to non-cross region image, folder: {folder_name}, path: {sample['non_cross']}")

        if original_img.shape != (128, 128):
            original_img = cv2.resize(original_img, (128, 128), interpolation=cv2.INTER_NEAREST)
        if cross_img.shape != (128, 128):
            cross_img = cv2.resize(cross_img, (128, 128), interpolation=cv2.INTER_NEAREST)
        if non_cross_img.shape != (128, 128):
            non_cross_img = cv2.resize(non_cross_img, (128, 128), interpolation=cv2.INTER_NEAREST)

        if label.shape[0] != 128 or label.shape[1] != 128:
            label_resized = np.zeros((128, 128, label.shape[2]), dtype=label.dtype)
            for c in range(label.shape[2]):
                label_resized[:, :, c] = cv2.resize(label[:, :, c], (128, 128), interpolation=cv2.INTER_NEAREST)
            label = label_resized

        original_tensor = torch.from_numpy(original_img).float().unsqueeze(0) / 255.0
        cross_tensor = torch.from_numpy(cross_img).float().unsqueeze(0) / 255.0
        non_cross_tensor = torch.from_numpy(non_cross_img).float().unsqueeze(0) / 255.0

        if is_train:
            # Reduced enhancement factor + small Gaussian noise for generalisation
            non_cross_tensor = non_cross_tensor * 1.15
            non_cross_tensor = non_cross_tensor + torch.randn_like(non_cross_tensor) * 0.01
            non_cross_tensor = torch.clamp(non_cross_tensor, 0, 1)

        label_tensor = torch.from_numpy(label).float().permute(2, 0, 1)
        return original_tensor, cross_tensor, non_cross_tensor, label_tensor, folder_name

    # ---------- Visualisation functions (kept from original) ----------
    def get_color_image(self, single_image, white_bg=True):
        if isinstance(single_image, torch.Tensor):
            single_image = single_image.detach().cpu().numpy()
        
        C, H, W = single_image.shape
        color_img = np.full((H, W, 3), 255, dtype=np.uint8) if white_bg else np.zeros((H, W, 3), dtype=np.uint8)
        for c in range(C):
            if c >= len(self.colors):
                continue
            mask = single_image[c] > 0.5
            color = self.colors[c]
            color_img[mask] = np.clip(color_img[mask] + color, 0, 255)
        return color_img

    def add_border(self, img, color, width):
        h, w = img.shape[:2]
        return cv2.rectangle(img.copy(), (0, 0), (w - 1, h - 1), color, thickness=width)

    def visualize_epoch_predictions(self, epoch):
        epoch_vis_dir = os.path.join(self.args.model_save_dir, f"epoch_{epoch}_visualization")
        if os.path.exists(epoch_vis_dir):
            shutil.rmtree(epoch_vis_dir)
        os.makedirs(epoch_vis_dir, exist_ok=True)

        test_dataset = self.val_loader.dataset
        total_samples = len(test_dataset)
        print(f"\n[Visualisation] Starting epoch {epoch} visualisation, processing {total_samples} samples, save path: {epoch_vis_dir}")

        self.model.eval()
        with torch.no_grad():
            for vis_idx, data_idx in enumerate(range(total_samples)):
                try:
                    sample = test_dataset[data_idx]
                    result = self.process_sample(sample, is_train=False)  # No enhancement for visualisation
                    if result is None:
                        continue
                    original_img, cross_img, non_cross_img, label, folder_name = result

                    original_batch = original_img.unsqueeze(0).to(self.device, non_blocking=True)
                    cross_batch = cross_img.unsqueeze(0).to(self.device, non_blocking=True)
                    non_cross_batch = non_cross_img.unsqueeze(0).to(self.device, non_blocking=True)
                    pred_logits, _ = self.model(cross_batch, non_cross_batch, original_batch)
                    pred_prob = torch.sigmoid(pred_logits)

                    original_np = original_img.squeeze().cpu().numpy()
                    original_rgb = cv2.cvtColor((original_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    original_rgb_target = cv2.resize(original_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)

                    cross_np = cross_img.squeeze().cpu().numpy()
                    cross_rgb = cv2.cvtColor((cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    cross_rgb_target = cv2.resize(cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)

                    non_cross_np = non_cross_img.squeeze().cpu().numpy()
                    non_cross_rgb = cv2.cvtColor((non_cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    non_cross_rgb_target = cv2.resize(non_cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)

                    label_np = label.cpu().numpy()
                    label_color = self.get_color_image(label_np, white_bg=True)
                    label_color_target = cv2.resize(label_color, self.target_size, interpolation=cv2.INTER_LINEAR)

                    pred_np = pred_prob.squeeze(0).cpu().numpy()
                    pred_color = self.get_color_image(pred_np, white_bg=True)
                    pred_color_target = cv2.resize(pred_color, self.target_size, interpolation=cv2.INTER_LINEAR)

                    original_with_border = self.add_border(original_rgb_target, self.border_color, self.border_width)
                    cross_with_border = self.add_border(cross_rgb_target, self.border_color, self.border_width)
                    non_cross_with_border = self.add_border(non_cross_rgb_target, self.border_color, self.border_width)
                    label_with_border = self.add_border(label_color_target, self.border_color, self.border_width)
                    prediction_with_border = self.add_border(pred_color_target, self.border_color, self.border_width)

                    save_paths = {
                        "original": os.path.join(epoch_vis_dir, f"{folder_name}_original_{self.target_size[0]}x{self.target_size[1]}.png"),
                        "cross": os.path.join(epoch_vis_dir, f"{folder_name}_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
                        "non_cross": os.path.join(epoch_vis_dir, f"{folder_name}_non_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
                        "label": os.path.join(epoch_vis_dir, f"{folder_name}_label_{self.target_size[0]}x{self.target_size[1]}.png"),
                        "prediction": os.path.join(epoch_vis_dir, f"{folder_name}_prediction_{self.target_size[0]}x{self.target_size[1]}.png")
                    }
                    for name, path in save_paths.items():
                        cv2.imwrite(path, locals()[name + "_with_border"])

                    if (vis_idx + 1) % 10 == 0 or vis_idx == total_samples - 1:
                        print(f"[Visualisation] Epoch {epoch} progress: {vis_idx + 1}/{total_samples} samples, current folder: {folder_name}")
                except Exception as e:
                    print(f"[Visualisation Error] Epoch {epoch} sample {folder_name} failed: {str(e)}")
                    continue
        print(f"[Visualisation] Epoch {epoch} visualisation completed, saved to: {epoch_vis_dir}\n")

    def save_visualizations(self):
        if os.path.exists(self.best_vis_dir):
            shutil.rmtree(self.best_vis_dir)
        os.makedirs(self.best_vis_dir, exist_ok=True)

        try:
            model_to_load = self._get_real_model()
            model_to_load.load_state_dict(torch.load(self.best_model_path))
            print(f"Loaded best model weights for visualisation: {self.best_model_path}")
        except Exception as e:
            print(f"Failed to load best model weights, using current model state for visualisation: {e}")

        test_dataset = self.val_loader.dataset
        total_samples = len(test_dataset)
        print(f"\n[Best Model Visualisation] Generating visualisations for {total_samples} validation samples, save path: {self.best_vis_dir}")

        self.model.eval()
        with torch.no_grad():
            for vis_idx, data_idx in enumerate(range(total_samples)):
                try:
                    sample = test_dataset[data_idx]
                    result = self.process_sample(sample, is_train=False)
                    if result is None:
                        continue
                    original_img, cross_img, non_cross_img, label, folder_name = result

                    original_batch = original_img.unsqueeze(0).to(self.device, non_blocking=True)
                    cross_batch = cross_img.unsqueeze(0).to(self.device, non_blocking=True)
                    non_cross_batch = non_cross_img.unsqueeze(0).to(self.device, non_blocking=True)
                    pred_logits, _ = self.model(cross_batch, non_cross_batch, original_batch)
                    pred_prob = torch.sigmoid(pred_logits)

                    original_np = original_img.squeeze().cpu().numpy()
                    original_rgb = cv2.cvtColor((original_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    original_rgb_target = cv2.resize(original_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)

                    cross_np = cross_img.squeeze().cpu().numpy()
                    cross_rgb = cv2.cvtColor((cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    cross_rgb_target = cv2.resize(cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)

                    non_cross_np = non_cross_img.squeeze().cpu().numpy()
                    non_cross_rgb = cv2.cvtColor((non_cross_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    non_cross_rgb_target = cv2.resize(non_cross_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)

                    label_np = label.cpu().numpy()
                    label_color = self.get_color_image(label_np, white_bg=True)
                    label_color_target = cv2.resize(label_color, self.target_size, interpolation=cv2.INTER_LINEAR)

                    pred_np = pred_prob.squeeze(0).cpu().numpy()
                    pred_color = self.get_color_image(pred_np, white_bg=True)
                    pred_color_target = cv2.resize(pred_color, self.target_size, interpolation=cv2.INTER_LINEAR)

                    original_with_border = self.add_border(original_rgb_target, self.border_color, self.border_width)
                    cross_with_border = self.add_border(cross_rgb_target, self.border_color, self.border_width)
                    non_cross_with_border = self.add_border(non_cross_rgb_target, self.border_color, self.border_width)
                    label_with_border = self.add_border(label_color_target, self.border_color, self.border_width)
                    prediction_with_border = self.add_border(pred_color_target, self.border_color, self.border_width)

                    save_paths = {
                        "original": os.path.join(self.best_vis_dir, f"{folder_name}_original_{self.target_size[0]}x{self.target_size[1]}.png"),
                        "cross": os.path.join(self.best_vis_dir, f"{folder_name}_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
                        "non_cross": os.path.join(self.best_vis_dir, f"{folder_name}_non_cross_{self.target_size[0]}x{self.target_size[1]}.png"),
                        "label": os.path.join(self.best_vis_dir, f"{folder_name}_label_{self.target_size[0]}x{self.target_size[1]}.png"),
                        "prediction": os.path.join(self.best_vis_dir, f"{folder_name}_prediction_{self.target_size[0]}x{self.target_size[1]}.png")
                    }
                    for name, path in save_paths.items():
                        cv2.imwrite(path, locals()[name + "_with_border"])

                    if (vis_idx + 1) % 10 == 0 or vis_idx == total_samples - 1:
                        print(f"[Best Model Visualisation] Progress: {vis_idx + 1}/{total_samples} samples, current folder: {folder_name}")
                except Exception as e:
                    print(f"[Best Model Visualisation Error] Sample {folder_name} failed: {str(e)}")
                    continue
        print(f"[Best Model Visualisation] All visualisations completed, saved to: {self.best_vis_dir}\n")

    def calculate_final_metrics(self):
        self.model.eval()
        all_p, all_r, all_f1 = [], [], []
        with torch.no_grad():
            for batch in self.val_loader:
                orig_list, cross_list, noncross_list, label_list = [], [], [], []
                for i in range(len(batch["original"])):
                    sample = {
                        "original": batch["original"][i],
                        "cross": batch["cross"][i],
                        "non_cross": batch["non_cross"][i],
                        "label": batch["label"][i],
                        "folder_name": batch["folder_name"][i]
                    }
                    res = self.process_sample(sample, is_train=False)
                    if res:
                        o, c, n, l, _ = res
                        orig_list.append(o)
                        cross_list.append(c)
                        noncross_list.append(n)
                        label_list.append(l)
                if not orig_list:
                    continue
                orig_batch = torch.stack(orig_list).to(self.device)
                cross_batch = torch.stack(cross_list).to(self.device)
                noncross_batch = torch.stack(noncross_list).to(self.device)
                label_batch = torch.stack(label_list).to(self.device)
                pred_logits, _ = self.model(cross_batch, noncross_batch, orig_batch)
                p, r, f1 = self.calculate_precision_recall_f1(pred_logits, label_batch)
                all_p.append(p)
                all_r.append(r)
                all_f1.append(f1)
        return np.mean(all_p), np.mean(all_r), np.mean(all_f1)

    # ---------- Data preparation ----------
    def prepare_data(self):
        print("Preparing data loaders")
        start_time = time.time()
        transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = FastStrokeDataset(self.args.data_dir, transform=transform,
                                          num_workers=self.args.num_workers, preload=True)
        val_dataset = FastStrokeDataset(self.args.val_data_dir, transform=transform,
                                        num_workers=self.args.num_workers, preload=True)

        train_size = len(train_dataset)
        val_size = len(val_dataset)
        print(f"Data preparation completed, training samples: {train_size}, validation samples: {val_size}, time taken: {time.time() - start_time:.2f}s")

        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=self.args.num_workers, pin_memory=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False,
                                     num_workers=self.args.num_workers // 2, pin_memory=True)
        print(f"Training data loader setup completed, batch size: {self.args.batch_size}, threads: {self.args.num_workers}")
        print(f"Validation data loader setup completed, batch size: {self.args.batch_size}, threads: {self.args.num_workers // 2}")
        print(f"Total data preparation time: {time.time() - start_time:.2f}s")
        return self.train_loader, self.val_loader

    # ---------- Main training loop with three metrics ----------
    def train(self):
        print("Starting SdaNet training with three metrics and spatial weighting")
        start_epoch = 1 if not self.args.continue_training else self.args.start_epoch
        total_epochs = self.args.epochs if not self.args.continue_training else self.args.start_epoch + self.args.num_epochs_continue - 1
        print(f"Starting epoch: {start_epoch}, Total training epochs: {total_epochs}")

        for epoch in range(start_epoch, total_epochs + 1):
            epoch_start = time.time()
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch}/{total_epochs} started, LR={current_lr:.8f}")
            print("=" * 60)

            # ---------- Training phase ----------
            self.model.train()
            train_total_loss = 0.0
            train_bce_loss = 0.0
            train_total_miou, train_cross_miou, train_nc_miou = 0.0, 0.0, 0.0
            train_p, train_r, train_f1 = 0.0, 0.0, 0.0
            batch_count = len(self.train_loader)

            for batch_idx, batch in enumerate(self.train_loader):
                orig_list, cross_list, noncross_list, label_list = [], [], [], []
                for i in range(len(batch["original"])):
                    sample = {
                        "original": batch["original"][i],
                        "cross": batch["cross"][i],
                        "non_cross": batch["non_cross"][i],
                        "label": batch["label"][i],
                        "folder_name": batch["folder_name"][i]
                    }
                    result = self.process_sample(sample, is_train=True)
                    if result is None:
                        continue
                    o, c, n, l, _ = result
                    orig_list.append(o)
                    cross_list.append(c)
                    noncross_list.append(n)
                    label_list.append(l)

                if len(orig_list) == 0:
                    continue

                orig_batch = torch.stack(orig_list).to(self.device)
                cross_batch = torch.stack(cross_list).to(self.device)
                noncross_batch = torch.stack(noncross_list).to(self.device)
                label_batch = torch.stack(label_list).to(self.device)

                self.optimizer.zero_grad()
                pred_logits, _ = self.model(cross_batch, noncross_batch, orig_batch)
                total_loss, bce_loss, _ = self.combined_loss(pred_logits, label_batch)
                total_miou, cross_miou, nc_miou = self.calculate_three_miou(pred_logits, label_batch)
                p, r, f1 = self.calculate_precision_recall_f1(pred_logits, label_batch)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_total_loss += total_loss.item()
                train_bce_loss += bce_loss.item()
                train_total_miou += total_miou
                train_cross_miou += cross_miou
                train_nc_miou += nc_miou
                train_p += p
                train_r += r
                train_f1 += f1

                if batch_idx % 10 == 0 or batch_idx == batch_count - 1:
                    print(f"[Train {batch_idx+1}/{batch_count}] Loss:{total_loss.item():.3f} | Total:{total_miou:.3f} | Cross:{cross_miou:.3f} | NonCross:{nc_miou:.3f} | P:{p:.3f} R:{r:.3f} F1:{f1:.3f}")

            train_total_loss /= batch_count
            train_bce_loss /= batch_count
            train_total_miou /= batch_count
            train_cross_miou /= batch_count
            train_nc_miou /= batch_count
            train_p /= batch_count
            train_r /= batch_count
            train_f1 /= batch_count

            print(f"[Train Done] Loss:{train_total_loss:.3f} | Total:{train_total_miou:.3f} Cross:{train_cross_miou:.3f} NonCross:{train_nc_miou:.3f} | P:{train_p:.3f} R:{train_r:.3f} F1:{train_f1:.3f}")

            # ---------- Validation phase ----------
            self.model.eval()
            val_total_loss = 0.0
            val_bce_loss = 0.0
            val_total_miou, val_cross_miou, val_nc_miou = 0.0, 0.0, 0.0
            val_p, val_r, val_f1 = 0.0, 0.0, 0.0
            val_batch_count = len(self.val_loader)

            with torch.no_grad():
                for batch_idx, batch in enumerate(self.val_loader):
                    orig_list, cross_list, noncross_list, label_list = [], [], [], []
                    for i in range(len(batch["original"])):
                        sample = {
                            "original": batch["original"][i],
                            "cross": batch["cross"][i],
                            "non_cross": batch["non_cross"][i],
                            "label": batch["label"][i],
                            "folder_name": batch["folder_name"][i]
                        }
                        result = self.process_sample(sample, is_train=False)
                        if result is None:
                            continue
                        o, c, n, l, _ = result
                        orig_list.append(o)
                        cross_list.append(c)
                        noncross_list.append(n)
                        label_list.append(l)

                    if len(orig_list) == 0:
                        continue

                    orig_batch = torch.stack(orig_list).to(self.device)
                    cross_batch = torch.stack(cross_list).to(self.device)
                    noncross_batch = torch.stack(noncross_list).to(self.device)
                    label_batch = torch.stack(label_list).to(self.device)

                    pred_logits, _ = self.model(cross_batch, noncross_batch, orig_batch)
                    total_loss, bce_loss, _ = self.combined_loss(pred_logits, label_batch)
                    total_miou, cross_miou, nc_miou = self.calculate_three_miou(pred_logits, label_batch)
                    p, r, f1 = self.calculate_precision_recall_f1(pred_logits, label_batch)

                    val_total_loss += total_loss.item()
                    val_bce_loss += bce_loss.item()
                    val_total_miou += total_miou
                    val_cross_miou += cross_miou
                    val_nc_miou += nc_miou
                    val_p += p
                    val_r += r
                    val_f1 += f1

                    if batch_idx % 10 == 0 or batch_idx == val_batch_count - 1:
                        print(f"[Val {batch_idx+1}/{val_batch_count}] Loss:{total_loss.item():.3f} | Total:{total_miou:.3f} Cross:{cross_miou:.3f} NonCross:{nc_miou:.3f} | P:{p:.3f} R:{r:.3f} F1:{f1:.3f}")

            val_total_loss /= val_batch_count
            val_bce_loss /= val_batch_count
            val_total_miou /= val_batch_count
            val_cross_miou /= val_batch_count
            val_nc_miou /= val_batch_count
            val_p /= val_batch_count
            val_r /= val_batch_count
            val_f1 /= val_batch_count

            print(f"[Val Done] Loss:{val_total_loss:.3f} | Total:{val_total_miou:.3f} Cross:{val_cross_miou:.3f} NonCross:{val_nc_miou:.3f} | P:{val_p:.3f} R:{val_r:.3f} F1:{val_f1:.3f}")

            self.scheduler.step()

            # Save best model based on Total mIoU
            if val_total_miou > self.best_val_iou:
                self.best_val_iou = val_total_miou
                self.save_model(epoch, val_total_loss, val_total_miou)
                self.save_best_info_with_metrics(val_total_miou, val_cross_miou, val_nc_miou, val_p, val_r, val_f1)

            # Visualisation
            if epoch % self.args.visualize_frequency == 0:
                self.visualize_epoch_predictions(epoch)

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch} finished in {epoch_time:.2f}s\n")

        # Post-training
        print("\n" + "="*60)
        final_p, final_r, final_f1 = self.calculate_final_metrics()
        print(f"[Final Metrics] Precision: {final_p:.4f} | Recall: {final_r:.4f} | F1: {final_f1:.4f}")
        print("Training completed, generating best model visualisations...")
        self.save_visualizations()
        print(f"Total training time: {time.time() - self.start_time:.2f}s")
        print(f"Best Total mIoU: {self.best_val_iou:.6f}")
        print("="*60)

    def save_model(self, epoch, val_loss, val_iou):
        try:
            model_to_save = self._get_real_model()
            torch.save(model_to_save.state_dict(), self.best_model_path)
            print(f"Best model saved: {self.best_model_path} (Epoch: {epoch}, Total mIoU: {val_iou:.6f})")
        except Exception as e:
            print(f"Model save failed: {e}")
            raise

    def save_best_info_with_metrics(self, total_miou, cross_miou, noncross_miou, precision, recall, f1):
        with open(self.best_info_path, 'w') as f:
            f.write("Best model metrics for paper:\n")
            f.write(f"Total mIoU: {total_miou:.6f}\n")
            f.write(f"Cross mIoU: {cross_miou:.6f}\n")
            f.write(f"Non-Cross mIoU: {noncross_miou:.6f}\n")
            f.write(f"Precision: {precision:.6f}\n")
            f.write(f"Recall: {recall:.6f}\n")
            f.write(f"F1-Score: {f1:.6f}\n")
            f.write(f"Class weights: {self.class_weights.cpu().numpy()}\n")
            f.write(f"LR: {self.optimizer.param_groups[0]['lr']:.8f}\n")
            f.write(f"Embed dim: {self.args.embed_dim}\n")
            f.write(f"Heads: {self.args.num_heads}\n")
            f.write(f"Classes: {self.args.num_stroke_classes}\n")
            f.write(f"Batch size: {self.args.batch_size}\n")
        print(f"Best info saved to: {self.best_info_path}")



def set_seed(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    
def main():
    parser = get_parser()
    args = parser.parse_args()
    set_seed(42)

    if not os.path.exists(args.val_data_dir):
        raise ValueError(f"Validation data directory does not exist: {args.val_data_dir}")
    if args.continue_training and args.pretrained_model_path == "":
        print("Error: --continue_training specified but --pretrained_model_path not provided")
        sys.exit(1)
    if len(args.class_weights.split(',')) != args.num_stroke_classes:
        raise ValueError(f"Number of class weights must match number of classes ({args.num_stroke_classes})")

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    print("="*60)
    print("SdaNet Training Configuration (Three Metrics + Spatial Weighting)")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Using GPU IDs: {args.gpu_ids}")
    print(f"Model save directory: {args.model_save_dir}")
    print(f"Class weights: {args.class_weights}")
    print(f"Visualize filtered samples: {args.vis_filtered_samples}")
    print(f"Save validation sample visualisation every {args.visualize_frequency} epochs")
    print("="*60)

    try:
        trainer = SdaNetTrainer(args)
        trainer.prepare_data()
        trainer.train()
    except Exception as e:
        print(f"Training process failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

