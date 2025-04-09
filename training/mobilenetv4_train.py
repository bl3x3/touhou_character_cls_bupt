import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision import transforms

# from torchvision.models import MobileNetV3
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import timm

# from torchsummary import summary

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.cuda.amp
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image, make_grid
import yaml
import csv
import os
from PIL import Image
import logging
from pathlib import Path
import warnings


def accuracy(output, target, topk=(1,)):
    """计算 Top-k 准确率"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # 转置后形状为 (maxk, batch_size)
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # 形状 (maxk, batch_size)

    res = []
    for k in topk:
        # 修改这里：正确指定sum()的参数
        correct_k = correct[:k].any(dim=0).sum()  # 移除了keepdim参数
        res.append(correct_k.float() * (100.0 / batch_size))  # 将乘法操作移到外面
    return res


def evaluate(model, val_loader, loss_func, device):
    model.eval()
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in tqdm(
            val_loader, desc="Evaluating", leave=False, total=len(val_loader)
        ):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_func(outputs, labels)
            total_loss += loss.item()
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            correct_top1 += acc1.item() * images.size(0)
            correct_top5 += acc5.item() * images.size(0)
            total_samples += images.size(0)

    avg_loss = total_loss / len(val_loader)
    avg_acc1 = correct_top1 / total_samples
    avg_acc5 = correct_top5 / total_samples
    return avg_loss, avg_acc1, avg_acc5


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    device="cpu",
    epochs=100,
    lr0=0.001,
    patience=8,  # epochs
    save_period=5,
    amp=True,  # 混合精度
    cos_lr=True,
    project="touhou_cls_mobilenetv4",
    name="mobilenetv4_01",
):
    # 保存args.yaml
    args = {
        "project": project,
        "name": name,
        "epochs": epochs,
        "lr0": lr0,
        "device": device,
        "train_batch": train_loader.batch_size,
        "val_batch": val_loader.batch_size,
        "patience": patience,
        "save_period": save_period,
        "amp": amp,
        "cos_lr": cos_lr,
    }
    output_dir = f"runs/{project}/{name}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Run tensorboard --logdir {output_dir} to view the training process")
    with open(os.path.join(output_dir, "args.yaml"), "w") as f:
        yaml.dump(args, f)

    writer = SummaryWriter(output_dir)
    scaler = torch.amp.GradScaler() if amp and device == "cuda" else None

    print(f"scaler: {scaler}")

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs) if cos_lr else None
    best_loss = float("inf")
    patience_counter = 0

    csv_path = os.path.join(output_dir, "results.csv")
    with open(csv_path, "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            ["epoch", "train/loss", "val/loss", "accuracy_top1", "accuracy_top5", "lr"]
        )

    try:
        for epoch in tqdm(range(epochs), desc=f"Training {name}"):
            model.train()
            total_train_loss = 0
            for i, (images, labels) in tqdm(
                enumerate(train_loader),
                desc=f"Epoch {epoch+1}/{epochs}",
                leave=False,
                total=len(train_loader),
            ):
                images, labels = images.to(device), labels.to(device)
                # print(type(images), type(labels))
                optimizer.zero_grad()  # 清空梯度
                if amp and device == "cuda":
                    # print("Using mixed precision training")
                    with torch.amp.autocast(device_type=device):
                        outputs = model(images)  # 前向传播
                        loss = loss_func(outputs, labels)  # 计算损失
                        scaler.scale(loss).backward()  # 反向传播
                        scaler.step(optimizer)  # 更新参数
                        scaler.update()
                else:
                    # print("Using standard training")
                    outputs = model(images)  # 前向传播
                    loss = loss_func(outputs, labels)  # 计算损失
                    loss.backward()  # 反向传播
                    optimizer.step()  # 更新参数

                total_train_loss += loss.item()
                writer.add_scalar(
                    "Loss/train_step", loss.item(), epoch * len(train_loader) + i
                )
            avg_train_loss = total_train_loss / len(train_loader)
            writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)

            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("LR", current_lr, epoch + 1)
            if scheduler:
                scheduler.step()  # 更新学习率

            # 保存训练过程中的图片
            if epoch == 0 or epoch == 1:
                save_image(
                    images, os.path.join(output_dir, f"train_batch{epoch+1}.jpg")
                )
                writer.add_image(
                    "Train Images", make_grid(images, nrow=8, normalize=True), epoch + 1
                )

            # 每个epoch进行一次验证
            print("Validation...")
            val_loss, val_acc1, val_acc5 = evaluate(
                model, val_loader, loss_func, device
            )
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val_top1", val_acc1, epoch)
            writer.add_scalar("Accuracy/val_top5", val_acc5, epoch)

            # 记录到CSV文件
            with open(csv_path, "a", newline="") as file:
                csv_writer = csv.writer(file)
                csv_writer.writerow(
                    [
                        epoch + 1,
                        avg_train_loss,
                        val_loss,
                        val_acc1,
                        val_acc5,
                        current_lr,
                    ]
                )

            # 输出到命令行
            print(
                f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Top-1 Acc: {val_acc1:.2f}%, Val Top-5 Acc: {val_acc5:.2f}%, LR: {current_lr:.6f}"
            )

            # 早停策略
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(
                    model.state_dict(), os.path.join(output_dir, f"{name}_best.pt")
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % save_period == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{name}_{epoch + 1}.pt"),
                )

            torch.save(model.state_dict(), os.path.join(output_dir, f"{name}_last.pt"))

    except Exception as e:
        print(f"An error occurred during training: {e}")
        # 保存模型参数和训练进度
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item() if "loss" in locals() else None,
            },
            os.path.join(output_dir, f"{name}_checkpoint_error.pth"),
        )
        print(
            f"Checkpoint saved to {os.path.join(output_dir, f'{name}_checkpoint_error.pth')}"
        )

    finally:
        writer.close()


class RobustImageFolder(Dataset):
    """健壮的图片数据集加载器，可以自动跳过损坏的图片"""
    
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.classes = []
        
        # 设置日志
        logging.basicConfig(
            filename=f'{self.root.name}_dataset_errors.log',
            level=logging.WARNING,
            format='%(asctime)s - %(message)s'
        )
        
        # 构建类别索引
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 收集所有有效的图片
        self._collect_valid_samples()
        
        if len(self.samples) == 0:
            raise RuntimeError(f"在{root}中没有找到有效的图片")
            
    def _collect_valid_samples(self):
        """收集所有可用的图片样本，跳过损坏的文件"""
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        skipped_files = 0
        total_files = 0
        
        # 首先计算总文件数
        for class_path in self.root.iterdir():
            if not class_path.is_dir():
                continue
            total_files += sum(1 for f in class_path.iterdir() if f.suffix.lower() in valid_extensions)
        
        # 使用tqdm显示进度
        pbar = tqdm(total=total_files, desc="验证图片文件")
        
        for class_path in self.root.iterdir():
            if not class_path.is_dir():
                continue
                
            class_idx = self.class_to_idx[class_path.name]
            
            for img_path in class_path.iterdir():
                if img_path.suffix.lower() not in valid_extensions:
                    continue
                    
                try:
                    # 尝试打开图片验证是否完整
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        with Image.open(img_path) as img:
                            img.verify()
                    # 再次打开以确保可以正确读取
                    with Image.open(img_path) as img:
                        img.load()
                    # 确保图片有正确的模式
                    if img.mode not in ['RGB', 'L']:
                        img = img.convert('RGB')
                        
                    self.samples.append((str(img_path), class_idx))
                    
                except Exception as e:
                    skipped_files += 1
                    logging.warning(f"跳过文件 {img_path}: {str(e)}")
                
                pbar.update(1)
        
        pbar.close()
        print(f"已加载 {len(self.samples)} 个有效图片，跳过 {skipped_files} 个无效文件")
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                return img, class_idx
        except Exception as e:
            # 如果在获取时出现问题，返回数据集中的下一个有效样本
            logging.error(f"加载 {img_path} 时出错: {str(e)}")
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)


if __name__ == "__main__":
    ####################
    # 在这里修改dataset路径
    train_root = "data/touhou_split/test"
    test_root = "data/touhou_split/val"
    ####################
    num_classes = len(os.listdir(train_root))

    print("Loading model...")
    model = timm.create_model("mobilenetv4_conv_medium.e500_r256_in1k", pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    #######################
    # 仅微调classifier层
    # 第5组注释掉下面井号框内的内容，取消注释框外后面的代码
    for name, param in model.named_parameters():
        if "classifier" not in name:  # 冻结除了最后一个线性层之外的所有参数
            param.requires_grad = False
        else:
            param.requires_grad = True
    #######################
    # require = False
    # for name, param in model.named_parameters():
    #     if "blocks.3" in name:
    #         require = True
    #     param.requires_grad = require

    print("Model loaded.")

    data_config = timm.data.resolve_model_data_config(model)
    #######################
    # 第4组和第5组注释掉下面井号框内的内容，取消注释框外后面两行代码
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomResizedCrop(
                size=(256, 256),
                scale=(0.75, 1.0),
                ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=(0.9, 1.1)),
            transforms.RandomRotation(degrees=45),
            transforms.RandomErasing(p=0.05, value="random"),
            # transforms.Normalize(mean=data_config["mean"], std=data_config["std"])
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                size=(269, 269), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(size=(256, 256)),
            # transforms.Normalize(mean=data_config["mean"], std=data_config["std"])
        ]
    )
    #######################
    # train_transform = timm.data.create_transform(**data_config, is_training=True)
    # test_transform = timm.data.create_transform(**data_config, is_training=False)

    print("Loading dataset...")
    train_dataset = RobustImageFolder(root=train_root, transform=train_transform)
    test_dataset = RobustImageFolder(root=test_root, transform=test_transform)

    # 在这里修改batch_size
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    print("Dataset loaded.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    lr = 1e-3
    epochs = 100
    patience = 5  # early stop
    loss_func = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=lr)

    model.to(device)       # 将模型移动到GPU上
    print(f"Training on {device}")
    train(
        model,
        train_loader,
        test_loader,
        optimizer,
        loss_func,
        device=device,
        epochs=epochs,
        patience=patience,
        name="mobilenetv4_01_02",      # 在这里指定实验名称
        amp=False,
    )
