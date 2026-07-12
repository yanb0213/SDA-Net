import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import re
import argparse
import time
import torch.nn.functional as F
from model.SCNet import SCNet
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

num_gpus = torch.cuda.device_count()
use_data_parallel = num_gpus >= 2
print(f"Found {num_gpus} GPUs (DataParallel: {use_data_parallel})")


def get_parser():
    parser = argparse.ArgumentParser(description='CSNet Training')
    # 原有参数保持不变
    parser.add_argument('--root_dir', type=str,
                        default="/home/wdxy/实验2/S-Dauble-Attention/dataset/stroke_cross/cross_hw/cross_train")
    parser.add_argument('--val_dir', type=str,
                        default="/home/wdxy/实验2/S-Dauble-Attention/dataset/stroke_cross/cross_hw/cross_test",
                        help='Validation directory')
    parser.add_argument('--in_channels', type=int, default=1)
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--continue_training', action='store_true', help='Continue from checkpoint')
    parser.add_argument('--pretrained_model_path', type=str, default="",
                        help='Pretrained model path')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--initial_learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--print_frequency', type=int, default=10, help='Print frequency for training')
    parser.add_argument('--rotation_range', type=int, default=15)
    parser.add_argument('--flip_prob', type=float, default=0.5)
    parser.add_argument('--translate_pixels', type=int, default=10, help='Max translation pixels')
    parser.add_argument('--scale_range', type=tuple, default=(0.9, 1.1))
    parser.add_argument('--model_save_dir', type=str,
                        default="/home/wdxy/实验2/S-Dauble-Attention/dataset/weight/CSNet_hw")
    parser.add_argument('--num_workers', type=int, default=32)

    # 新增参数：学习率减半的间隔（每多少个epoch）
    parser.add_argument('--lr_decay_epochs', type=int, default=1,
                        help='Number of epochs to decay learning rate by half (default: 10)')
    return parser


def find_image_folders(root_dir):
    """Find image folders with pattern image_*"""
    image_folders = []
    pattern = re.compile(r'image_\d+')
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            if pattern.fullmatch(dir_name):
                image_folders.append(os.path.join(root, dir_name))
    print(f"Found {len(image_folders)} image folders in {root_dir}")
    return image_folders


class StrokeCrossDataset(Dataset):
    """Dataset for stroke cross detection"""

    def __init__(self, root_dir, transform=None):
        self.transform = transform  # 变换函数（接收PIL图像）
        self.input_files = []
        self.label_files = []
        image_folders = find_image_folders(root_dir)
        valid_pairs = 0
        for folder in image_folders:
            for file in os.listdir(folder):
                if file.endswith(('union.png', 'union.jpg', 'union.jpeg')):
                    input_path = os.path.join(folder, file)
                    label_file = file.replace("union", "intersection")
                    label_path = os.path.join(folder, label_file)
                    if os.path.exists(label_path):
                        self.input_files.append(input_path)
                        self.label_files.append(label_path)
                        valid_pairs += 1
        print(f"Found {valid_pairs} valid pairs")

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        try:
            input_path = self.input_files[idx]
            label_path = self.label_files[idx]

            if not os.path.exists(input_path) or not os.path.exists(label_path):
                print(f"Error: Files not found - Input: {input_path}, Label: {label_path}")
                return None, None

            # 加载为PIL灰度图（保持PIL格式进行变换）
            input_img = Image.open(input_path).convert('L')
            label_img = Image.open(label_path).convert('L')

            if input_img is None or label_img is None:
                print(f"Error: Image loading failed - Input: {input_path}, Label: {label_path}")
                return None, None

            # 对PIL图像应用变换（此时fill参数生效）
            if self.transform:
                # 同时对输入和标签应用相同的空间变换（保证对齐）
                input_img = self.transform(input_img)
                label_img = self.transform(label_img)

            # 转换为Tensor并归一化（输入归一化到[0,1]，标签转为0/1）
            input_tensor = transforms.ToTensor()(input_img)  # 自动归一化到[0,1]
            label_np = np.array(label_img).astype(np.float32) / 255.0  # 标签0/255 -> 0/1
            label_tensor = torch.from_numpy(label_np).unsqueeze(0)  # 增加通道维度

            return input_tensor, label_tensor
        except Exception as e:
            print(f"Error processing files: {input_path}, {label_path} - {e}")
            return None, None


def calculate_iou(pred, target, threshold=0.5):
    """Calculate IoU between prediction and target"""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (intersection / union).item()


def calculate_accuracy(pred, target, threshold=0.5):
    """Calculate accuracy between prediction and target"""
    pred = (pred > threshold).float()
    correct = (pred == target).float().sum()
    total = torch.numel(pred)
    return (correct / total).item()


def validate_model(model, val_dataloader, device):
    """Validate model performance"""
    model.eval()
    total_iou = 0.0
    total_accuracy = 0.0
    count = 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            if inputs is None or labels is None:
                continue

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            for pred, target in zip(outputs, labels):
                iou = calculate_iou(pred, target)
                accuracy = calculate_accuracy(pred, target)
                total_iou += iou
                total_accuracy += accuracy
                count += 1

    if count == 0:
        return 0.0, 0.0
    return total_iou / count, total_accuracy / count


class CSNetTrainer:
    """CSNet training class"""

    def __init__(self, args):
        self.args = args
        self.best_iou = 0.0
        self.best_model_path = None
        self.start_time = time.time()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = SCNet(args.in_channels, args.base_channels).to(self.device)
        if use_data_parallel:
            self.model = nn.DataParallel(self.model)
            print(f"Using {num_gpus} GPUs with DataParallel")

        # Set up loss function and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.initial_learning_rate)

        # 学习率调度器：按固定间隔减半（替换原来的ReduceLROnPlateau）
        # step_size=args.lr_decay_epochs：每多少个epoch减半
        # gamma=0.5：每次调整为当前学习率 × 0.5（即减半）
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.lr_decay_epochs,
            gamma=0.5,
        )

        # Create model save directory
        os.makedirs(args.model_save_dir, exist_ok=True)

    def train_epoch(self, train_dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        batch_count = 0
        valid_batches = 0

        current_lr = self.optimizer.param_groups[0]['lr']

        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            if inputs is None or labels is None:
                print(f"Error: Invalid batch {batch_idx}")
                continue

            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Handle size mismatch
            if outputs.shape != labels.shape:
                outputs = F.interpolate(outputs, size=labels.shape[2:], mode='bilinear', align_corners=True)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            valid_batches += 1

            if batch_idx % self.args.print_frequency == 0:
                avg_loss = running_loss / valid_batches
                print(f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(train_dataloader)} | "
                      f"Valid batches: {valid_batches} | Loss: {loss.item():.4f} | Avg loss: {avg_loss:.4f} | "
                      f"LR: {current_lr:.8f}")

        return running_loss / valid_batches if valid_batches > 0 else float('inf')

    def validate(self, val_dataloader):
        """Validate model"""
        val_iou, val_accuracy = validate_model(self.model, val_dataloader, self.device)
        return val_iou, val_accuracy

    def save_best_model(self, epoch, val_iou, val_accuracy):
        """Save best model"""
        if val_iou > self.best_iou:
            self.best_iou = val_iou
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

            # Save best model
            best_path = os.path.join(self.args.model_save_dir, f"csnet_best_iou_{val_iou:.4f}_epoch_{epoch + 1}.pth")
            torch.save(model_to_save.state_dict(), best_path)
            self.best_model_path = best_path

            print(f"Saved new best model! Best IoU: {val_iou:.4f}, Accuracy: {val_accuracy:.4f}")
            print(f"Model saved to: {best_path}")

    def train(self, train_dataloader, val_dataloader):
        """Main training loop"""
        print(f"Starting CSNet training...")
        print(f"Total epochs: {self.args.num_epochs}")
        print(f"Initial learning rate: {self.args.initial_learning_rate}")
        print(f"Learning rate decay: every {self.args.lr_decay_epochs} epochs (halve)")  # 打印衰减间隔
        print(f"Batch size: {self.args.batch_size}")

        for epoch in range(self.args.start_epoch, self.args.start_epoch + self.args.num_epochs):
            epoch_start = time.time()

            # Training
            train_loss = self.train_epoch(train_dataloader, epoch)

            # Validation
            val_iou, val_accuracy = self.validate(val_dataloader)

            # 学习率调整：每个epoch结束后触发（StepLR无需传入指标）
            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"\nEpoch {epoch + 1} Summary | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val IoU: {val_iou:.4f} | "
                  f"Val Accuracy: {val_accuracy:.4f} | "
                  f"Current LR: {current_lr:.8f} | "  # 显示当前学习率
                  f"Time: {epoch_time:.2f}s")
            print("-" * 80)

            # Save best model
            self.save_best_model(epoch, val_iou, val_accuracy)

        # Final summary
        final_info = {
            'best_iou': self.best_iou,
            'best_model_path': self.best_model_path,
            'total_epochs': self.args.num_epochs,
            'total_time': time.time() - self.start_time
        }

        print(f"\nTraining completed!")
        print(f"Best IoU: {self.best_iou:.4f}")
        print(f"Best model saved to: {self.best_model_path}")
        print(f"Total training time: {time.time() - self.start_time:.2f}s")

        return final_info


def load_pretrained_model(model, pretrained_path, device):
    """Load pretrained model"""
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained model from {pretrained_path}...")
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)

            # Handle DataParallel checkpoint
            new_state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith('module.'):
                    new_state_dict[key[7:]] = value
                else:
                    new_state_dict[key] = value

            model.load_state_dict(new_state_dict, strict=False)
            print("Pretrained model loaded successfully")
        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
    else:
        print("No pretrained model provided, starting from scratch")


def main(args):
    """Main function"""
    # Print parameters
    print("\n===== CSNet Training Parameters =====")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print(f"Available GPUs: {num_gpus}")
    print("========================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize trainer
    trainer = CSNetTrainer(args)

    # Load pretrained model if specified
    if args.continue_training:
        load_pretrained_model(trainer.model, args.pretrained_model_path, device)

    # 数据变换（调整为128x128节省显存）
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(args.rotation_range, fill=255),
        transforms.RandomHorizontalFlip(args.flip_prob),
        transforms.RandomVerticalFlip(args.flip_prob),
        transforms.RandomAffine(
            degrees=0,
            translate=(args.translate_pixels / 128, args.translate_pixels / 128),
            scale=args.scale_range,
            fill=255
        ),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
    ])

    # Create datasets
    print("Creating datasets...")
    train_dataset = StrokeCrossDataset(args.root_dir, train_transforms)
    val_dataset = StrokeCrossDataset(args.val_dir, val_transforms)

    if len(train_dataset) == 0:
        print("Error: No training data found!")
        return
    if len(val_dataset) == 0:
        print("Error: No validation data found!")
        return

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers // 2,
        drop_last=False,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Dataloader workers: {args.num_workers}")

    # Start training
    try:
        trainer.train(train_dataloader, val_dataloader)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Create model save directory
    os.makedirs(args.model_save_dir, exist_ok=True)

    main(args)
