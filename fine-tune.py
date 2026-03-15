import os
import cv2
import sys
import random
import torch
import numpy
import torch.nn as nn
import torch.nn.functional as functional
from pathlib import Path
import multiprocessing
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import albumentations
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


SINET_ROOT = Path(__file__).resolve().parent / "SINet-V2"
sys.path.insert(0, str(SINET_ROOT))
from lib.Network_Res2Net_GRA_NCD import Network
from utils.utils import clip_gradient

@dataclass
class Config:
    pretrained_ckpt_path: str = "./pretrained.pth"
    data_root:            str = "./ACD1K"
    best_model_path:      str = "./sinetv2_fine_tuned.pth"
    onnx_path:            str = "./sinetv2_fine_tuned.onnx"
    input_h:              int = 352
    input_w:              int = 352
    batch_size:           int = 36
    num_epochs:           int = 80
    learning_rate:      float = 1e-4
    grad_clip:          float = 0.5
    num_workers:        int   = 4
    channel:            int   = 32
    opset:              int   = 18
    seed:               int   = 24
config = Config()

def set_seed(seed: int):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SegmentationDataset(Dataset):
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, image_dir, mask_dir, transform=None, mode="train"):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.mode = mode
        if self.mode not in {"train", "val"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        image_map = {
            p.stem: p for p in self.image_dir.iterdir()
            if p.suffix.lower() in self.extensions
        }
        mask_map = {
            p.stem: p for p in self.mask_dir.iterdir()
            if p.suffix.lower() in self.extensions
        }
        common = sorted(set(image_map) & set(mask_map))
        if not common:
            raise RuntimeError("no matching image-mask pairs found")
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
            raise RuntimeError(f"failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"failed to read mask: {mask_path}")
        if image.shape[:2] != mask.shape[:2]:
            raise RuntimeError(f"mismatched image/mask size for '{img_path.name}'")
        raw_mask = mask.astype(numpy.float32) / 255.0
        orig_h, orig_w = raw_mask.shape
        if self.transform is not None:
            aug = self.transform(image=image, mask=raw_mask)
            image = aug["image"]
            transformed_mask = aug["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            transformed_mask = torch.from_numpy(raw_mask)
        if self.mode == "train":
            if not torch.is_tensor(transformed_mask):
                transformed_mask = torch.from_numpy(transformed_mask)
            transformed_mask = transformed_mask.float()
            if transformed_mask.ndim == 2:
                transformed_mask = transformed_mask.unsqueeze(0)
            return image, transformed_mask
        original_mask = torch.from_numpy(raw_mask).unsqueeze(0).float()
        return image, original_mask, (orig_h, orig_w)

class randomCrop(albumentations.DualTransform):
    def __init__(self, border=30, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.border = border

    def apply(self, img, crop_coords=(), **params):
        x1, y1, x2, y2 = crop_coords
        return img[y1:y2, x1:x2]

    def apply_to_mask(self, mask, crop_coords=(), **params):
        x1, y1, x2, y2 = crop_coords
        return mask[y1:y2, x1:x2]

    def get_params_dependent_on_data(self, params, data):
        img = data["image"]
        h, w = img.shape[:2]
        min_w = max(1, w - self.border)
        min_h = max(1, h - self.border)
        crop_w = numpy.random.randint(min_w, w + 1)
        crop_h = numpy.random.randint(min_h, h + 1)
        x1 = (w - crop_w) >> 1
        y1 = (h - crop_h) >> 1
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        return {"crop_coords": (x1, y1, x2, y2)}
    def get_transform_init_args_names(self):
        return ("border",)
    
train_transform = albumentations.Compose([
    albumentations.HorizontalFlip(p=0.5),
    randomCrop(border=30, p=0.2),
    albumentations.Resize(config.input_h, config.input_w, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
    albumentations.Rotate(limit=15, interpolation=cv2.INTER_CUBIC, mask_interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, p=0.2),
    albumentations.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.0, 2.0), hue=0.0, p=0.2),
    albumentations.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.5), p=0.2),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
val_transform = albumentations.Compose([
    albumentations.Resize(config.input_h, config.input_w),
    albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

@torch.no_grad()
def normalize_prediction_map(prob: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    b = prob.shape[0]
    flat = prob.view(b, -1)
    minv = flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    maxv = flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    return (prob - minv) / (maxv - minv + eps)


@torch.no_grad()
def mae_from_logits_per_sample(
    logits: torch.Tensor,
    original_masks: list[torch.Tensor],
    original_sizes: list[tuple[int, int]],
    eps: float = 1e-8,
) -> torch.Tensor:
    maes = []
    for i, (target, (orig_h, orig_w)) in enumerate(zip(original_masks, original_sizes)):
        pred = logits[i:i+1]
        prob = torch.sigmoid(pred)
        if prob.shape[-2:] != (orig_h, orig_w):
            prob = functional.interpolate(
                prob,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )
        prob = normalize_prediction_map(prob, eps=eps)
        target = target.to(prob.device, non_blocking=True)
        mae = torch.abs(prob - target).mean()
        maes.append(mae)
    return torch.stack(maes, dim=0)

def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020) w deprecation cleanup
    """
    weit = 1 + 5 * torch.abs(
        functional.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = functional.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()
    
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

def val_collate_fn(batch):
    images = torch.stack([x[0] for x in batch], dim=0)
    original_masks = [x[1] for x in batch]
    original_sizes = [x[2] for x in batch]
    return images, original_masks, original_sizes
    
def main():
    set_seed(config.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    device = torch.device("cuda")

    base_model = Network(channel=config.channel, imagenet_pretrained=True)
    load_pretrained(base_model, config.pretrained_ckpt_path)
    base_model = base_model.to(device)

    train_dataset = SegmentationDataset(image_dir=Path(config.data_root) / "train/images", mask_dir=Path(config.data_root) / "train/masks", transform=train_transform, mode="train")
    val_dataset  = SegmentationDataset(image_dir=Path(config.data_root) / "val/images", mask_dir=Path(config.data_root) / "val/masks", transform=val_transform, mode="val")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=val_collate_fn
    )

    optimizer = torch.optim.Adam(base_model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=40,
        gamma=0.1,
    )
    best_mae = float("inf")
    for epoch in range(1, config.num_epochs + 1):
        base_model.train()
        running_train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.num_epochs} [train]")
        for images, masks in train_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            preds = base_model(images)
            if not isinstance(preds, (list, tuple)) or len(preds) != 4:
                raise RuntimeError(f"Expected 4 outputs from SINet-V2, got {type(preds)} with len={len(preds) if isinstance(preds, (list, tuple)) else 'n/a'}")
            loss_init = sum(structure_loss(preds[i], masks) for i in range(3))
            loss_final = structure_loss(preds[3], masks)
            total_loss = loss_init + loss_final
            total_loss.backward()
            clip_gradient(optimizer, config.grad_clip)
            optimizer.step()
            running_train_loss += float(total_loss.detach().cpu())
            train_bar.set_postfix(loss=f"{total_loss.item():.4f}")
        base_model.eval()
        val_mae_sum = 0.0
        val_image_count = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{config.num_epochs} [val]")
            for images, original_masks, original_sizes in val_bar:
                images = images.to(device, non_blocking=True)
                outputs = base_model(images)
                pred = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
                batch_mae_tensor = mae_from_logits_per_sample(
                    pred,
                    original_masks,
                    original_sizes,
                )
                batch_mae_sum = batch_mae_tensor.sum().item()
                batch_size_actual = batch_mae_tensor.numel()
                val_mae_sum += batch_mae_sum
                val_image_count += batch_size_actual
                batch_mae = batch_mae_tensor.mean().item()
                running_mean_mae = val_mae_sum / max(val_image_count, 1)
                val_bar.set_postfix(
                    batch_mae=f"{batch_mae:.4f}",
                    val_mae=f"{running_mean_mae:.4f}",
                )
        mean_mae = val_mae_sum / max(val_image_count, 1)
        scheduler.step()
        if mean_mae < best_mae:
            best_mae = mean_mae
            torch.save(base_model.state_dict(), config.best_model_path)
    print("training finished")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()