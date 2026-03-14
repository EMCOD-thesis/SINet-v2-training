import os
import cv2
import sys
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import multiprocessing as mp
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import onnx
import onnxruntime as ort

SINET_ROOT = Path(__file__).resolve().parent / "SINet-V2"
sys.path.insert(0, str(SINET_ROOT))
from lib.Network_Res2Net_GRA_NCD import Network

@dataclass
class Config:
    pretrained_ckpt_path: str = "./pretrained.pth"
    data_root:            str = "./ACD1K"
    best_model_path:      str = "./sinetv2_fine_tuned.pth"
    onnx_path:            str = "./sinetv2_fine_tuned.onnx"
    input_h:              int = 320
    input_w:              int = 320
    batch_size:           int = 36
    num_epochs:           int = 100
    learning_rate:      float = 1e-4
    weight_decay:       float = 1e-4
    grad_clip_norm:     float = 0.5
    num_workers:        int   = 4
    channel:            int   = 32
    opset:              int   = 18
    seed:               int   = 23
    amp:                bool  = True
    eval_threshold:     float = 0.5
config = Config()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SegmentationDataset(Dataset):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        image_map = {}
        for p in self.image_dir.iterdir():
            if p.suffix.lower() in self.extensions:
                image_map[p.stem] = p
        mask_map = {}
        for p in self.mask_dir.iterdir():
            if p.suffix.lower() in self.extensions:
                mask_map[p.stem] = p

        common = sorted(set(image_map) & set(mask_map))
        if not common:
            raise RuntimeError("No matching image-mask pairs found")
        self.samples = [(image_map[s], mask_map[s]) for s in common]
        missing_masks = set(image_map) - set(mask_map)
        missing_images = set(mask_map) - set(image_map)
        if missing_masks:
            print(f"{len(missing_masks)} images without masks")
        if missing_images:
            print(f"{len(missing_images)} masks without images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        if image.shape[:2] != mask.shape[:2]:
            print(
                f"[WARN] Resizing mask to match image for sample '{img_path.name}': "
                f"image={image.shape[:2]}, mask={mask.shape[:2]}"
            )
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return image, mask.float()
    
train_transform = A.Compose([
    A.Resize(config.input_h, config.input_w),
    A.HorizontalFlip(p=0.4),
    A.VerticalFlip(p=0.1),
    A.Affine(
        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
        scale=(0.85, 1.15),
        rotate=(-15, 15),
        shear=(-5, 5),
        interpolation=cv2.INTER_LINEAR,
        mask_interpolation=cv2.INTER_NEAREST,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.2,
    ),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.GaussNoise(std_range=(0.02, 0.06), p=1.0),
    ], p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Resize(config.input_h, config.input_w),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

@torch.no_grad()
def segmentation_metrics_from_logits(logits, target, threshold=0.5, eps=1e-6):
    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    mae = torch.abs(prob - target).mean(dim=(1, 2, 3))
    return {
        "iou": iou.mean().item(),
        "dice": dice.mean().item(),
        "mae": mae.mean().item(),
    }

@torch.no_grad()
def soft_dice_from_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (denom + eps)
    return dice.mean().item()


def multi_output_structure_loss(outputs, masks):
    if isinstance(outputs, (list, tuple)):
        return sum(structure_loss(o, masks) for o in outputs)
    return structure_loss(outputs, masks)

def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    pasted from SINet-v2/myTrain_Val.py w small bugfix
    """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

class FinalOutputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            return out[-1]
        return out
    
def load_pretrained(model: nn.Module, ckpt_path: str):
    if not ckpt_path:
        raise RuntimeError("No pretrained checkpoint path")

    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"Pretrained checkpoint not found at: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    cleaned = {}
    for k, v in state_dict.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v

    model.load_state_dict(cleaned, strict=True)
    
def main():
    set_seed(config.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    device = torch.device("cuda")

    base_model = Network(channel=config.channel, imagenet_pretrained=False)
    load_pretrained(base_model, config.pretrained_ckpt_path)
    base_model = base_model.to(device)

    train_dataset = SegmentationDataset(image_dir=Path(config.data_root) / "train/images", mask_dir=Path(config.data_root) / "train/masks", transform=train_transform)
    val_dataset  = SegmentationDataset(image_dir=Path(config.data_root) / "val/images", mask_dir=Path(config.data_root) / "val/masks", transform=val_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, base_model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=50,
        gamma=0.1,
    )
    scaler = GradScaler("cuda", enabled=config.amp)

    best_score = -float("inf")
    for epoch in range(1, config.num_epochs + 1):
        base_model.train()
        running_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs} [train]")
        for images, masks in train_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=config.amp):
                outputs = base_model(images)
                total_loss = multi_output_structure_loss(outputs, masks)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(base_model.parameters(), max_norm=config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            running_train_loss += float(total_loss.detach().cpu())
            train_bar.set_postfix(loss=f"{total_loss.item():.4f}")
        avg_train_loss = running_train_loss / max(len(train_loader), 1)

        base_model.eval()
        running_val_total_loss = 0.0
        running_val_final_loss = 0.0
        val_iou  = []
        val_dice = []
        val_mae  = []
        val_soft_dice = []
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{config.num_epochs} [val]")
            for images, masks in val_bar:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                outputs = base_model(images)
                pred = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
                val_total_loss = multi_output_structure_loss(outputs, masks)
                val_final_loss = structure_loss(pred, masks)
                running_val_total_loss += float(val_total_loss.detach().cpu())
                running_val_final_loss += float(val_final_loss.detach().cpu())
                m = segmentation_metrics_from_logits(pred, masks, threshold=config.eval_threshold)
                s_dice = soft_dice_from_logits(pred, masks)
                val_iou.append(m["iou"])
                val_dice.append(m["dice"])
                val_mae.append(m["mae"])
                val_soft_dice.append(s_dice)
                val_bar.set_postfix(final_loss=f"{val_final_loss.item():.4f}", soft_dice=f"{s_dice:.4f}", mae=f"{m['mae']:.4f}",)
        avg_val_total_loss = running_val_total_loss / max(len(val_loader), 1)
        avg_val_final_loss = running_val_final_loss / max(len(val_loader), 1)
        mean_iou = float(np.mean(val_iou)) if val_iou else 0.0
        mean_dice = float(np.mean(val_dice)) if val_dice else 0.0
        mean_mae = float(np.mean(val_mae)) if val_mae else 1.0
        mean_soft_dice = float(np.mean(val_soft_dice)) if val_soft_dice else 0.0
        val_score = mean_soft_dice - mean_mae
        scheduler.step()
        if val_score > best_score:
            best_score = val_score
            torch.save(base_model.state_dict(), config.best_model_path)
    print("training finished")

    base_model.load_state_dict(torch.load(config.best_model_path, map_location="cpu"))
    base_model.eval()
    base_model.cpu()
    export_model = FinalOutputWrapper(base_model).eval()
    dummy_input = torch.randn(1, 3, config.input_h, config.input_w, dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            export_model,
            dummy_input,
            config.onnx_path,
            export_params=True,
            opset_version=config.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["mask_logits"],
            dynamic_axes=None,
        )
    onnx_model = onnx.load(config.onnx_path)
    onnx.checker.check_model(onnx_model)
    ort_session = ort.InferenceSession(
        config.onnx_path,
        providers=["CPUExecutionProvider"]
    )
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    print("ONNX model checked and inference tested successfully")
    print("Output shape:", ort_outputs[0].shape)

if __name__ == "__main__":
    mp.freeze_support()
    main()