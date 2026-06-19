"""训练咀嚼/进食声音分类模型。

这个脚本负责完成以下工作：
1. 扫描音频数据集。
2. 按类别和录音组划分训练集、验证集、测试集。
3. 将音频转换为 Mel 频谱。
4. 使用 CNN / 残差 CNN 模型进行分类训练。
5. 保存最优模型、最终模型、训练历史和评估指标。
"""

import argparse  # 读取命令行参数，例如训练轮数、batch size、学习率等。
import json  # 保存 class_to_idx、history、metrics 等 JSON 文件。
import os  # 判断当前系统，例如 Windows 下默认 num_workers 设置为 0 更稳定。
import random  # 用于随机打乱、随机裁剪和数据增强。
from collections import defaultdict  # 用于按类别、录音组整理样本。
from pathlib import Path  # 用 pathlib 处理路径，比普通字符串路径更清晰。

import torch  # PyTorch 主库，负责张量计算、GPU 加速和模型训练。
import torchaudio  # PyTorch 音频库，用于读取音频、重采样、生成 Mel 频谱。
from torch import nn  # 神经网络模块，例如卷积层、全连接层、损失函数。
from torch.utils.data import DataLoader, Dataset  # Dataset 和 DataLoader 用于封装、批量读取数据。


# 支持读取的音频格式。扫描数据集时，只会把这些后缀的文件当作音频样本。
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def parse_args():
    """解析命令行参数。

    用户可以在运行脚本时通过参数控制数据路径、训练轮数、模型大小、是否开启混合精度等。
    """
    parser = argparse.ArgumentParser(description="Train a GPU-accelerated chewing sound food classifier.")

    # 基础训练参数。
    parser.add_argument("--data-dir", type=Path, default=Path("archive/clips_rd"))  # 默认数据集目录。
    parser.add_argument("--output-dir", type=Path, default=Path("runs/chewing_audio"))  # 默认训练输出目录。
    parser.add_argument("--epochs", type=int, default=50)  # 训练轮数。
    parser.add_argument("--batch-size", type=int, default=32)  # 每个 batch 的音频数量。
    parser.add_argument("--lr", type=float, default=3e-4)  # 学习率。
    parser.add_argument("--weight-decay", type=float, default=1e-4)  # 权重衰减，用于减轻过拟合。
    parser.add_argument("--sample-rate", type=int, default=16000)  # 统一采样率。
    parser.add_argument("--duration", type=float, default=4.0)  # 每条音频统一长度，单位秒。
    parser.add_argument("--n-mels", type=int, default=96)  # Mel 频谱通道数。
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 4)  # Windows 下多进程读取容易出错，默认设为 0。
    parser.add_argument("--seed", type=int, default=42)  # 随机种子，便于复现实验。
    parser.add_argument("--train-ratio", type=float, default=0.8)  # 训练集比例。
    parser.add_argument("--val-ratio", type=float, default=0.1)  # 验证集比例，剩余部分作为测试集。
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision training.")  # 开启 CUDA 混合精度，加速并降低显存占用。
    parser.add_argument("--limit-per-class", type=int, default=0, help="Use only N clips per class for a quick test. 0 means all clips.")  # 快速测试用，每类只取 N 条。

    # 下面是为了提高准确率和训练速度新增的参数。
    parser.add_argument(
        "--model-size",
        choices=["legacy", "tiny", "base", "large"],
        default="base",
        help="legacy is the original small CNN; base/large are stronger residual CNNs.",
    )  # legacy 是旧版小模型；tiny/base/large 是新版残差模型。
    parser.add_argument("--dropout", type=float, default=0.35)  # 分类器 dropout，防止过拟合。
    parser.add_argument("--label-smoothing", type=float, default=0.05)  # 标签平滑，让模型不要过度自信。
    parser.add_argument("--scheduler", choices=["cosine", "onecycle"], default="onecycle")  # 学习率调度器。
    parser.add_argument("--patience", type=int, default=20, help="Early stop after N epochs without val improvement. 0 disables it.")  # 早停轮数。
    parser.add_argument("--min-epochs", type=int, default=25, help="Do not early stop before this many epochs.")  # 早停前至少训练多少轮。
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm. 0 disables it.")  # 梯度裁剪，避免梯度爆炸。
    parser.add_argument("--cache-audio", action="store_true", help="Load processed audio into RAM to reduce disk I/O and speed up epochs.")  # 把音频缓存到内存，提高训练速度。
    parser.add_argument("--compile", action="store_true", help="Try torch.compile for faster training on supported PyTorch versions.")  # 尝试 PyTorch 2.x 编译加速。
    parser.add_argument("--no-specaugment", action="store_true", help="Disable SpecAugment on Mel spectrograms during training.")  # 关闭频谱增强。
    return parser.parse_args()


def set_seed(seed):
    """设置随机种子，让数据划分和训练结果尽量稳定。"""
    random.seed(seed)  # Python 随机种子。
    torch.manual_seed(seed)  # PyTorch CPU 随机种子。
    torch.cuda.manual_seed_all(seed)  # PyTorch CUDA 随机种子。


def list_audio_files(data_dir, limit_per_class=0):
    """扫描数据集目录，返回音频样本列表和类别映射。

    数据集格式要求：data_dir/类别名/音频文件。
    返回的 samples 每一项为：(音频路径, 数字标签, 类别名)。
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # 每个子文件夹代表一个类别，按名称排序保证类别编号稳定。
    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class folders found under: {data_dir}")

    class_to_idx = {path.name: idx for idx, path in enumerate(class_dirs)}
    samples = []
    for class_dir in class_dirs:
        # 递归扫描当前类别文件夹下的所有音频文件。
        files = sorted(
            path for path in class_dir.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
        # 快速测试时只取每个类别前 N 条。
        if limit_per_class > 0:
            files = files[:limit_per_class]
        label = class_to_idx[class_dir.name]
        samples.extend((path, label, class_dir.name) for path in files)

    if not samples:
        raise RuntimeError(f"No audio files found under: {data_dir}")
    return samples, class_to_idx


def group_key(path):
    """从文件名中提取录音组编号，减少数据泄漏。

    例如 fries_9_88.wav 会把 9 作为组编号。
    同一组切片会被划分到同一个 train/val/test 集合，避免同一段原始音频同时出现在训练和测试里。
    """
    parts = path.stem.split("_")
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
        return parts[-2]
    return path.stem


def split_samples(samples, train_ratio, val_ratio, seed):
    """按照类别和录音组划分训练集、验证集、测试集。"""
    rng = random.Random(seed)
    by_class = defaultdict(lambda: defaultdict(list))

    # 先按类别、录音组归类。
    for path, label, class_name in samples:
        by_class[label][group_key(path)].append((path, label, class_name))

    train, val, test = [], [], []
    for label, groups in by_class.items():
        group_items = list(groups.values())
        rng.shuffle(group_items)

        n_groups = len(group_items)
        train_groups = max(1, int(round(n_groups * train_ratio)))
        val_groups = max(1, int(round(n_groups * val_ratio))) if n_groups >= 3 else 0

        # 如果类别录音组太少，保证训练集至少有数据，并尽量保留验证集和测试集。
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
    """自定义音频数据集。

    负责读取音频、转单声道、重采样、幅度归一化、裁剪/补零、训练增强和可选缓存。
    """

    def __init__(self, samples, sample_rate, duration, training=False, cache_audio=False):
        self.samples = samples
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)  # 固定输入长度，例如 16000 * 4 秒。
        self.training = training
        self.cache_audio = cache_audio
        self.cached_waveforms = None

        # cache_audio 用于加速训练：第一次初始化时把音频读入内存，后续 epoch 不再重复读硬盘。
        if cache_audio:
            self.cached_waveforms = []
            print(f"Caching {len(samples)} audio clips into RAM...")
            for path, _, _ in samples:
                # 缓存时使用中心裁剪，训练阶段仍会在 __getitem__ 中做随机增强。
                self.cached_waveforms.append(self._load_waveform(path, random_crop=False).cpu())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, _ = self.samples[index]

        # 如果开启缓存，直接从内存里取；否则现场读取音频文件。
        if self.cached_waveforms is not None:
            waveform = self.cached_waveforms[index].clone()
        else:
            waveform = self._load_waveform(path, random_crop=self.training)

        # 只有训练集做数据增强，验证集和测试集保持稳定。
        if self.training:
            waveform = self._augment(waveform)

        return waveform, torch.tensor(label, dtype=torch.long)

    def _load_waveform(self, path, random_crop):
        """读取并预处理一条音频。"""
        waveform, source_rate = torchaudio.load(str(path))  # waveform 形状通常是 [声道数, 采样点数]。
        waveform = waveform.mean(dim=0, keepdim=True)  # 多声道转单声道。

        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, self.sample_rate)

        # 幅度归一化，减少不同录音音量差异对模型的影响。
        peak = waveform.abs().max().clamp_min(1e-6)
        waveform = waveform / peak
        waveform = self._crop_or_pad(waveform, random_crop=random_crop)
        return waveform

    def _crop_or_pad(self, waveform, random_crop=False):
        """把音频裁剪或补零到固定长度。"""
        length = waveform.shape[-1]
        if length > self.num_samples:
            # 训练时随机裁剪，评估时中心裁剪。
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
        """训练阶段的波形增强：音量扰动、加噪声、时间平移。"""
        gain = random.uniform(0.70, 1.30)  # 随机调整音量。
        waveform = waveform * gain

        if random.random() < 0.40:
            noise = torch.randn_like(waveform) * random.uniform(0.001, 0.008)  # 添加轻微随机噪声。
            waveform = waveform + noise

        if random.random() < 0.30:
            max_shift = max(1, int(waveform.shape[-1] * 0.05))  # 最多平移 5% 的长度。
            shift = random.randint(-max_shift, max_shift)
            waveform = torch.roll(waveform, shifts=shift, dims=-1)

        return waveform.clamp(-1.0, 1.0)


class ConvBNAct(nn.Sequential):
    """卷积 + 批归一化 + SiLU 激活的组合模块。"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class SqueezeExcite(nn.Module):
    """SE 注意力模块。

    作用：让模型自动学习哪些通道更重要，从而提升特征表达能力。
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 对每个通道做全局平均池化。
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),  # 输出 0~1 的通道权重。
        )

    def forward(self, x):
        return x * self.net(x)


class ResidualBlock(nn.Module):
    """残差块。

    残差连接可以让更深的网络更容易训练，通常比普通堆叠卷积更稳定。
    """

    def __init__(self, in_channels, out_channels, stride=1, drop=0.0):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SqueezeExcite(out_channels)
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

        # 如果通道数或尺寸发生变化，需要用 1x1 卷积调整 shortcut 的形状。
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
    """音频分类模型。

    输入：原始音频波形 [batch, 1, samples]
    处理：波形 -> Mel 频谱 -> dB -> 标准化 -> CNN 特征 -> 分类结果
    输出：每个类别的 logits 分数。
    """

    def __init__(self, num_classes, sample_rate, n_mels, model_size="base", dropout=0.35, specaugment=True):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.model_size = model_size
        self.specaugment = specaugment

        # MelSpectrogram 把一维音频波形转换成二维“声音图片”。
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            f_min=40,
            f_max=sample_rate // 2,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")  # 转换到分贝尺度。

        # legacy 是旧模型；tiny/base/large 使用更强的残差特征提取器。
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
        """构建旧版普通 CNN，方便兼容之前训练出的模型。"""
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
        """构建新版残差 CNN。

        tiny/base/large 的区别主要是通道数不同；large 更强但显存和时间消耗更大。
        """
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
        """对每条音频的 Mel 频谱做标准化，减小音量和录音条件差异。"""
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def _specaugment(self, x):
        """SpecAugment 频谱增强。

        训练时随机遮挡一段频率和一段时间，让模型不要只记住局部特征，提高泛化能力。
        """
        if not (self.training and self.specaugment):
            return x

        _, _, freq_bins, time_steps = x.shape

        # 随机遮挡部分频率通道。
        if freq_bins > 8 and random.random() < 0.80:
            width = random.randint(2, max(2, min(freq_bins // 6, 18)))
            start = random.randint(0, max(0, freq_bins - width))
            x = x.clone()
            x[:, :, start : start + width, :] = 0

        # 随机遮挡部分时间帧。
        if time_steps > 8 and random.random() < 0.80:
            width = random.randint(4, max(4, min(time_steps // 5, 48)))
            start = random.randint(0, max(0, time_steps - width))
            if not x.is_leaf:
                x = x.clone()
            x[:, :, :, start : start + width] = 0

        return x


def make_loader(samples, args, training):
    """创建 DataLoader。"""
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
    """根据训练集类别数量计算类别权重，缓解类别不均衡问题。"""
    labels = torch.tensor([label for _, label, _ in samples], dtype=torch.long)
    counts = torch.bincount(labels, minlength=num_classes).float()
    return counts.sum() / (counts.clamp_min(1.0) * num_classes)


def unwrap_model(model):
    """如果模型被 torch.compile 包装过，则取回原始模型，便于保存权重。"""
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def run_epoch(model, loader, criterion, optimizer, scaler, scheduler, device, training, use_amp, grad_clip):
    """运行一轮训练或评估，返回平均 loss 和准确率。"""
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    use_amp = use_amp and device.type == "cuda"  # 只有 CUDA 环境才启用混合精度。

    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(waveforms)
                loss = criterion(logits, labels)

            # 训练阶段才需要反向传播和参数更新。
            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)  # 裁剪前先反缩放梯度。
                        torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), grad_clip)
                    optimizer.step()

                # OneCycleLR 需要每个 batch 更新一次学习率。
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
    """把 Python 数据保存为 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def save_checkpoint(path, model, args, class_to_idx, epoch, val_acc=None, test_acc=None):
    """保存模型 checkpoint。

    checkpoint 中除了模型参数，还保存采样率、音频长度、Mel 参数、模型大小等信息，方便预测脚本重建模型。
    """
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
    """主训练流程。"""
    args = parse_args()
    set_seed(args.seed)

    # 自动选择训练设备：优先使用 CUDA GPU。
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # 输入尺寸固定时可以加速卷积。
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Training will run on CPU.")

    # 读取数据并划分 train/val/test。
    samples, class_to_idx = list_audio_files(args.data_dir, args.limit_per_class)
    train_samples, val_samples, test_samples = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    # 保存类别映射和数据划分摘要。
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

    # 创建模型。
    model = AudioCNN(
        num_classes=len(class_to_idx),
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        model_size=args.model_size,
        dropout=args.dropout,
        specaugment=not args.no_specaugment,
    ).to(device)

    # 可选 torch.compile 加速；不支持时自动跳过。
    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile enabled.")
        except Exception as exc:
            print(f"torch.compile failed, continuing without it: {exc}")

    # 使用类别权重和标签平滑的交叉熵损失。
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(train_samples, len(class_to_idx)).to(device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 创建学习率调度器。
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

        # CosineAnnealingLR 按 epoch 更新；OneCycleLR 已经在 batch 内更新。
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

        # 验证集准确率提升时保存 best_model.pt。
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(best_path, model, args, class_to_idx, epoch, val_acc=val_acc)
        else:
            epochs_without_improvement += 1

        # 每一轮都保存 history，防止训练中断后没有记录。
        save_json(args.output_dir / "history.json", history)

        # 早停：如果验证集长期没有提升，则提前结束训练，节省时间。
        if args.patience > 0 and epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
            print(f"Early stopping: validation accuracy did not improve for {args.patience} epochs.")
            break

    # 加载验证集最优模型，并在测试集上做最终评估。
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
