import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GPU-accelerated chewing sound food classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("archive/clips_rd"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/chewing_audio"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--n-mels", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision training.")
    parser.add_argument("--limit-per-class", type=int, default=0, help="Use only N clips per class for a quick test. 0 means all clips.")

    parser.add_argument(
        "--model-size",
        choices=["legacy", "tiny", "base", "large"],
        default="base",
        help="legacy is the original small CNN; base/large are stronger residual CNNs.",
    )
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--scheduler", choices=["cosine", "onecycle"], default="onecycle")
    parser.add_argument("--patience", type=int, default=20, help="Early stop after N epochs without val improvement. 0 disables it.")
    parser.add_argument("--min-epochs", type=int, default=25, help="Do not early stop before this many epochs.")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm. 0 disables it.")
    parser.add_argument("--cache-audio", action="store_true", help="Load processed audio into RAM to reduce disk I/O and speed up epochs.")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile for faster training on supported PyTorch versions.")
    parser.add_argument("--no-specaugment", action="store_true", help="Disable SpecAugment on Mel spectrograms during training.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_audio_files(data_dir, limit_per_class=0):
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class folders found under: {data_dir}")

    class_to_idx = {path.name: idx for idx, path in enumerate(class_dirs)}
    samples = []
    for class_dir in class_dirs:
        files = sorted(
            path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
        if limit_per_class > 0:
            files = files[:limit_per_class]
        label = class_to_idx[class_dir.name]
        samples.extend((path, label, class_dir.name) for path in files)

    if not samples:
        raise RuntimeError(f"No audio files found under: {data_dir}")
    return samples, class_to_idx


def group_key(path):
    parts = path.stem.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return parts[-2]
    return path.stem


def split_samples(samples, train_ratio, val_ratio, seed):
    rng = random.Random(seed)
    by_class = defaultdict(lambda: defaultdict(list))
    for path, label, class_name in samples:
        by_class[label][group_key(path)].append((path, label, class_name))

    train, val, test = [], [], []
    for label, groups in by_class.items():
        group_items = list(groups.values())
        rng.shuffle(group_items)

        n_groups = len(group_items)
        train_groups = max(1, int(round(n_groups * train_ratio)))
        val_groups = max(1, int(round(n_groups * val_ratio))) if n_groups >= 3 else 0
        if train_groups + val_groups >= n_groups:
            train_groups = max(1, n_groups - 2)
            val_groups = 1 if n_groups >= 2 else 0

        selected_train = group_items[:train_groups]
        selected_val = group_items[train_groups : train_groups + val_groups]
        selected_test = group_items[train_groups + val_groups :]

        train.extend(sample for group in selected_train for sample in group)
        val.extend(sample for group in selected_val for sample in group)
        test.extend(sample for group in selected_test for sample in group)

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


class ChewingSoundDataset(Dataset):
    def __init__(self, samples, sample_rate, duration, training=False, cache_audio=False):
        self.samples = samples
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        self.training = training
        self.cache_audio = cache_audio
        self.cached_waveforms = None

        if cache_audio:
            self.cached_waveforms = []
            print(f"Caching {len(samples)} audio clips into RAM...")
            for path, _, _ in samples:
                self.cached_waveforms.append(self._load_waveform(path, random_crop=False).cpu())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, _ = self.samples[index]
        if self.cached_waveforms is not None:
            waveform = self.cached_waveforms[index].clone()
        else:
            waveform = self._load_waveform(path, random_crop=self.training)

        if self.training:
            waveform = self._augment(waveform)

        return waveform, torch.tensor(label, dtype=torch.long)

    def _load_waveform(self, path, random_crop):
        waveform, source_rate = torchaudio.load(str(path))
        waveform = waveform.mean(dim=0, keepdim=True)

        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, self.sample_rate)

        peak = waveform.abs().max().clamp_min(1e-6)
        waveform = waveform / peak
        waveform = self._crop_or_pad(waveform, random_crop=random_crop)
        return waveform

    def _crop_or_pad(self, waveform, random_crop=False):
        length = waveform.shape[-1]
        if length > self.num_samples:
            if random_crop:
                start = random.randint(0, length - self.num_samples)
            else:
                start = (length - self.num_samples) // 2
            waveform = waveform[:, start : start + self.num_samples]
        elif length < self.num_samples:
            pad = self.num_samples - length
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform

    @staticmethod
    def _augment(waveform):
        gain = random.uniform(0.70, 1.30)
        waveform = waveform * gain

        if random.random() < 0.40:
            noise = torch.randn_like(waveform) * random.uniform(0.001, 0.008)
            waveform = waveform + noise

        if random.random() < 0.30:
            max_shift = max(1, int(waveform.shape[-1] * 0.05))
            shift = random.randint(-max_shift, max_shift)
            waveform = torch.roll(waveform, shifts=shift, dims=-1)

        return waveform.clamp(-1.0, 1.0)


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, drop=0.0):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SqueezeExcite(out_channels)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.drop(x)
        return self.act(x + residual)


class AudioCNN(nn.Module):
    def __init__(self, num_classes, sample_rate, n_mels, model_size="base", dropout=0.35, specaugment=True):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.model_size = model_size
        self.specaugment = specaugment
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            f_min=40,
            f_max=sample_rate // 2,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

        if model_size == "legacy":
            self.features, feature_dim = self._build_legacy_features()
        else:
            self.features, feature_dim = self._build_residual_features(model_size)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )

    @staticmethod
    def _build_legacy_features():
        features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.10),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.20),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        return features, 256

    @staticmethod
    def _build_residual_features(model_size):
        channel_map = {
            "tiny": [32, 64, 128, 192],
            "base": [48, 96, 192, 256],
            "large": [64, 128, 256, 384],
        }
        channels = channel_map[model_size]
        c1, c2, c3, c4 = channels
        features = nn.Sequential(
            ConvBNAct(1, c1, kernel_size=3, stride=1),
            ResidualBlock(c1, c1, stride=1, drop=0.05),
            ResidualBlock(c1, c2, stride=2, drop=0.08),
            ResidualBlock(c2, c2, stride=1, drop=0.08),
            ResidualBlock(c2, c3, stride=2, drop=0.12),
            ResidualBlock(c3, c3, stride=1, drop=0.12),
            ResidualBlock(c3, c4, stride=2, drop=0.16),
            ResidualBlock(c4, c4, stride=1, drop=0.16),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        return features, c4

    def forward(self, waveform):
        x = self.mel(waveform)
        x = self.to_db(x)
        x = self._normalize_logmel(x)
        x = self._specaugment(x)
        x = self.features(x)
        return self.classifier(x)

    @staticmethod
    def _normalize_logmel(x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def _specaugment(self, x):
        if not (self.training and self.specaugment):
            return x

        _, _, freq_bins, time_steps = x.shape
        if freq_bins > 8 and random.random() < 0.80:
            width = random.randint(2, max(2, min(freq_bins // 6, 18)))
            start = random.randint(0, max(0, freq_bins - width))
            x = x.clone()
            x[:, :, start : start + width, :] = 0

        if time_steps > 8 and random.random() < 0.80:
            width = random.randint(4, max(4, min(time_steps // 5, 48)))
            start = random.randint(0, max(0, time_steps - width))
            if not x.is_leaf:
                x = x.clone()
            x[:, :, :, start : start + width] = 0

        return x


def make_loader(samples, args, training):
    dataset = ChewingSoundDataset(
        samples=samples,
        sample_rate=args.sample_rate,
        duration=args.duration,
        training=training,
        cache_audio=args.cache_audio,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )


def class_weights(samples, num_classes):
    labels = torch.tensor([label for _, label, _ in samples], dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    return counts.sum() / (counts.clamp_min(1.0) * num_classes)


def unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def run_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, training, use_amp, grad_clip):
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    use_amp = use_amp and device.type == "cuda"

    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(waveforms)
                loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), grad_clip)
                    optimizer.step()
                if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size

    if total_seen == 0:
        return 0.0, 0.0
    return total_loss / total_seen, total_correct / total_seen


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def save_checkpoint(path, model, args, class_to_idx, epoch, val_acc=None, test_acc=None):
    payload = {
        "epoch": epoch,
        "model_state": unwrap_model(model).state_dict(),
        "class_to_idx": class_to_idx,
        "sample_rate": args.sample_rate,
        "duration": args.duration,
        "n_mels": args.n_mels,
        "model_size": args.model_size,
        "dropout": args.dropout,
        "specaugment": not args.no_specaugment,
        "args": vars(args),
    }
    if val_acc is not None:
        payload["val_acc"] = val_acc
    if test_acc is not None:
        payload["test_acc"] = test_acc
    torch.save(payload, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training will run on CPU.")

    samples, class_to_idx = list_audio_files(args.data_dir, args.limit_per_class)
    train_samples, val_samples, test_samples = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "class_to_idx.json", class_to_idx)
    save_json(
        args.output_dir / "split_summary.json",
        {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
            "classes": idx_to_class,
        },
    )

    train_loader = make_loader(train_samples, args, training=True)
    val_loader = make_loader(val_samples, args, training=False)
    test_loader = make_loader(test_samples, args, training=False)

    model = AudioCNN(
        num_classes=len(class_to_idx),
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        model_size=args.model_size,
        dropout=args.dropout,
        specaugment=not args.no_specaugment,
    ).to(device)

    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled.")
        except Exception as exc:
            print(f"torch.compile failed, continuing without it: {exc}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_samples, len(class_to_idx)).to(device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.epochs,
            steps_per_epoch=max(1, len(train_loader)),
            pct_start=0.15,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Classes: {len(class_to_idx)}")
    print(f"Samples: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    print(
        f"Model: {args.model_size} | amp={args.amp} | cache_audio={args.cache_audio} | "
        f"scheduler={args.scheduler} | label_smoothing={args.label_smoothing}"
    )

    history = []
    best_val_acc = 0.0
    best_path = args.output_dir / "best_model.pt"
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            scheduler,
            device,
            training=True,
            use_amp=args.amp,
            grad_clip=args.grad_clip,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            scaler,
            None,
            device,
            training=False,
            use_amp=args.amp,
            grad_clip=args.grad_clip,
        )

        if args.scheduler == "cosine":
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.2e}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, args, class_to_idx, epoch, val_acc=val_acc)
        else:
            epochs_without_improvement += 1

        save_json(args.output_dir / "history.json", history)

        if args.patience > 0 and epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
            print(f"Early stopping: validation accuracy did not improve for {args.patience} epochs.")
            break

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(checkpoint["model_state"])
    test_loss, test_acc = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer,
        scaler,
        None,
        device,
        training=False,
        use_amp=args.amp,
        grad_clip=args.grad_clip,
    )

    save_checkpoint(args.output_dir / "last_model.pt", model, args, class_to_idx, checkpoint["epoch"], test_acc=test_acc)
    save_json(args.output_dir / "history.json", history)
    save_json(
        args.output_dir / "metrics.json",
        {
            "best_val_acc": best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "best_model": str(best_path),
            "model_size": args.model_size,
        },
    )

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")
    print(f"Saved best model to: {best_path}")


if __name__ == "__main__":
    main()
