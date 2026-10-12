
## -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  
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
                        default="/media/text/afedbba0-93b4-490d-9d12-20e5b7b22aa7/yan/exp2/S-Dauble-Attention/dataset/strock_semantics/semantics_hw/semantics_train",
                        help='Training data directory')
    parser.add_argument('--val_data_dir', type=str,
                        default="/media/text/afedbba0-93b4-490d-9d12-20e5b7b22aa7/yan/exp2/S-Dauble-Attention/dataset/strock_semantics/semantics_hw/semantics_val",
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
                        default="/media/text/afedbba0-93b4-490d-9d12-20e5b7b22aa7/yan/exp2/S-Dauble-Attention/dataset/weight/hw_ioucross",
                        help='Model save directory')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')

    # GPU settings
    parser.add_argument('--gpu_ids', type=str, default="0,1", help='GPU IDs to use, e.g., "0,1"')

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

#        xiugai               
    parser.add_argument('--feat_scale', type=float, default=1.0,
                        help='Scale factor for encoder channels (for param-matched ablation)')
    parser.add_argument('--no_double_attention', action='store_true', default=False,
                        help='Disable double attention module (for param-matched ablation)')                    
                        

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
#        xiugai
#        print(f"Initializing SdaNet")
#        self.model = SdaNet(
#            embed_dim=args.embed_dim,
#            num_heads=args.num_heads,
#            num_stroke_classes=args.num_stroke_classes,
#            encoder_mode='D', 
#            use_skip_connections=use_skip,
#        ).to(self.device)
#        print("Model initialization completed")

        print(f"Initializing SdaNet")
        self.model = SdaNet(
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_stroke_classes=args.num_stroke_classes,
            encoder_mode='D', 
            use_skip_connections=use_skip,
            feat_scale=args.feat_scale,
            use_double_attention=not args.no_double_attention
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
#    def calculate_three_miou(self, pred_logits, label, eps=1e-6):
#        pred = (torch.sigmoid(pred_logits) > 0.5).float()
#        B, C, H, W = pred.shape
#    
#        total_list, cross_list, noncross_list = [], [], []
#    
#        for b in range(B):
#            p = pred[b]          # [C, H, W]
#            t = label[b]         # [C, H, W]
#            sum_t = t.sum(dim=0) # [H, W]
#    
#            cross_mask = (sum_t >= 2).float()
#            noncross_mask = (sum_t <= 1).float()
#    
#            # ---------- Total mIoU ----------
#            t_ious = []
#            for c in range(C):
#                if t[c].sum() > 0:  
#                    inter = (p[c] * t[c]).sum()
#                    union = (p[c] + t[c]).sum() - inter
#                    iou = (inter + eps) / (union + eps)
#                    t_ious.append(iou.item())
#            total_miou = sum(t_ious) / len(t_ious) if t_ious else 1.0
#            total_list.append(total_miou)
#    
#            # ---------- Cross mIoU ----------
#            c_ious = []
#            for c in range(C):
#                t_cross = t[c] * cross_mask
#                if t_cross.sum() > 0:  
#                    inter = (p[c] * t[c] * cross_mask).sum()
#                    union = ((p[c] + t[c]) * cross_mask).sum() - inter
#                    iou = (inter + eps) / (union + eps)
#                    c_ious.append(iou.item())
#            cross_miou = sum(c_ious) / len(c_ious) if c_ious else 1.0
#            cross_list.append(cross_miou)
#    
#            # ---------- Non?Cross mIoU ----------
#            n_ious = []
#            for c in range(C):
#                t_nc = t[c] * noncross_mask
#                if t_nc.sum() > 0:   
#                    inter = (p[c] * t[c] * noncross_mask).sum()
#                    union = ((p[c] + t[c]) * noncross_mask).sum() - inter
#                    iou = (inter + eps) / (union + eps)
#                    n_ious.append(iou.item())
#            noncross_miou = sum(n_ious) / len(n_ious) if n_ious else 1.0
#            noncross_list.append(noncross_miou)
#    
#        return np.mean(total_list), np.mean(cross_list), np.mean(noncross_list)

    def calculate_three_miou(self, pred_logits, label, eps=1e-6):
        pred = (torch.sigmoid(pred_logits) > 0.5).float()
        B, C, H, W = pred.shape
    
        total_list = []          
        noncross_list = []       
    
        total_cross_pixels = 0
        correct_cross_pixels = 0
    
        for b in range(B):
            p = pred[b]         
            t = label[b]         
    
            sum_t = t.sum(dim=0) 
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
    
            # ---------- Non-Cross mIoU  ----------
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
    
            
            pred_equal = (p == t).all(dim=0)  # [H, W] bool
            cross_pixels = cross_mask.sum().item()
            if cross_pixels > 0:
                
                tp_cross = (pred_equal & cross_mask.bool()).sum().item()
                correct_cross_pixels += tp_cross
                total_cross_pixels += cross_pixels
    
        total_miou = np.mean(total_list)
        noncross_miou = np.mean(noncross_list)
    
        cross_miou = correct_cross_pixels / (total_cross_pixels + eps) if total_cross_pixels > 0 else 1.0
    
        return total_miou, cross_miou, noncross_miou
    

    # ---------- Loss functions with spatial weighting for non-cross pixels ----------
    def weighted_bce_loss(self, pred_logits, target, lambda_adapt=1.0):
        
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        class_weights = self.class_weights.view(1, -1, 1, 1)
    
        p = torch.sigmoid(pred_logits)
        error = torch.abs(p - target)                     # (B, K, H, W)
        w_adapt = 1.0 + lambda_adapt * error.mean(dim=1, keepdim=True)  # (B, 1, H, W)
    
        weighted_bce = bce * class_weights * w_adapt
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
            raise ValueError(f"Failed to load cross region image, folder: {folder_name}, path: {sample['cross']}")
        if non_cross_img is None:
            raise ValueError(f"Failed to load non-cross region image, folder: {folder_name}, path: {sample['non_cross']}")
    
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
           
            cross_tensor = cross_tensor * 1.15
            cross_tensor = cross_tensor + torch.randn_like(cross_tensor) * 0.01
            cross_tensor = torch.clamp(cross_tensor, 0, 1)
    
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
