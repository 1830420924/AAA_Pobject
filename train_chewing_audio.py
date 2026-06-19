import argparse  # 导入 argparse：用来从命令行读取参数，比如训练轮数、batch 大小、数据路径等。
import json  # 导入 json：用来把类别映射、训练历史、最终指标保存成 .json 文件，方便之后查看。
import os  # 导入 os：用来判断当前系统类型，比如 Windows 上 DataLoader 多进程默认设为 0 更稳。
import random  # 导入 random：用来做随机打乱、随机裁剪、简单音频增强等操作。
from collections import defaultdict  # 导入 defaultdict：创建“自动带默认值”的字典，后面按类别和录音组整理样本会用到。
from pathlib import Path  # 导入 Path：比字符串路径更好用，可以跨平台处理文件和文件夹路径。

import torch  # 导入 PyTorch 主库：负责张量计算、GPU 加速、模型训练等核心功能。
import torchaudio  # 导入 torchaudio：PyTorch 官方音频库，用来读取 wav、重采样、生成 Mel 频谱。
from torch import nn  # 从 torch 中导入 nn：nn 里包含神经网络层，比如卷积、全连接、损失函数等。
from torch.utils.data import DataLoader, Dataset  # 导入 Dataset 和 DataLoader：用于封装数据集和按批次读取数据。


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}  # 定义允许读取的音频后缀名集合，脚本只会把这些文件当作音频样本。


def parse_args():  # 定义命令行参数解析函数，运行脚本时可以通过 --epochs 之类的参数修改配置。
    parser = argparse.ArgumentParser(  # 创建参数解析器对象，它会负责理解用户在命令行输入的各种参数。
        description="Train a GPU-accelerated chewing sound food classifier."  # 设置脚本说明文字，用户执行 -h 时会看到。
    )  # 结束 ArgumentParser 的创建。
    parser.add_argument("--data-dir", type=Path, default=Path("archive/clips_rd"))  # 数据集目录，默认读取当前项目里的 archive/clips_rd。
    parser.add_argument("--output-dir", type=Path, default=Path("runs/chewing_audio"))  # 训练结果保存目录，模型和指标都会写到这里。
    parser.add_argument("--epochs", type=int, default=50)  # 训练轮数，默认 50 轮，符合你提出的训练要求。
    parser.add_argument("--batch-size", type=int, default=32)  # 每次送进模型的音频数量，显存不足时可以改小，比如 16 或 8。
    parser.add_argument("--lr", type=float, default=3e-4)  # 学习率，控制模型参数每次更新的步子大小。
    parser.add_argument("--weight-decay", type=float, default=1e-4)  # 权重衰减，用来轻微抑制过拟合，相当于给模型复杂度加约束。
    parser.add_argument("--sample-rate", type=int, default=16000)  # 统一采样率，所有音频都会转成 16000Hz，便于模型处理。
    parser.add_argument("--duration", type=float, default=4.0)  # 每条音频统一截取或补齐到 4 秒，保证输入长度一致。
    parser.add_argument("--n-mels", type=int, default=96)  # Mel 频谱的频率维度数量，可以理解为把声音切成 96 个频率刻度。
    parser.add_argument("--num-workers", type=int, default=0 if os.name == "nt" else 4)  # DataLoader 读取数据的子进程数量，Windows 默认 0 更不容易出兼容问题。
    parser.add_argument("--seed", type=int, default=42)  # 随机种子，用来让数据划分和随机操作尽量可复现。
    parser.add_argument("--train-ratio", type=float, default=0.8)  # 训练集比例，默认 80% 的录音组用于训练。
    parser.add_argument("--val-ratio", type=float, default=0.1)  # 验证集比例，默认 10% 的录音组用于验证。
    parser.add_argument(  # 添加一个可选开关参数，用来手动打开混合精度训练。
        "--amp",  # 命令行写 --amp 时表示开启 Automatic Mixed Precision，也就是混合精度。
        action="store_true",  # store_true 表示：只要命令行出现 --amp，这个参数就会变成 True。
        help="Enable CUDA mixed precision. Leave off if loss becomes nan.",  # 参数帮助文字：如果 loss 出现 nan，就不要开混合精度。
    )  # 结束 --amp 参数定义。
    parser.add_argument(  # 添加快速测试用参数，可以限制每个类别最多读取多少条音频。
        "--limit-per-class",  # 命令行参数名，例如 --limit-per-class 40 表示每类只取 40 条。
        type=int,  # 参数类型是整数。
        default=0,  # 默认 0 表示不限制，使用每个类别下的全部音频。
        help="Use only N clips per class for a quick smoke test. 0 means all clips.",  # 参数帮助文字，说明这个参数适合快速冒烟测试。
    )  # 结束 --limit-per-class 参数定义。
    return parser.parse_args()  # 真正解析命令行参数，并把结果返回给主程序使用。


def set_seed(seed):  # 定义设置随机种子的函数，让随机划分和训练过程尽量稳定。
    random.seed(seed)  # 设置 Python 内置 random 模块的随机种子。
    torch.manual_seed(seed)  # 设置 PyTorch CPU 随机种子。
    torch.cuda.manual_seed_all(seed)  # 设置所有 CUDA GPU 的随机种子；没有 GPU 时这行也不会影响训练。


def list_audio_files(data_dir, limit_per_class=0):  # 扫描数据集目录，返回所有音频文件路径和类别编号。
    if not data_dir.exists():  # 如果用户给的数据目录不存在，就不能继续训练。
        raise FileNotFoundError(f"Data directory not found: {data_dir}")  # 抛出清楚的错误，告诉用户哪个目录没找到。

    class_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())  # 找到数据目录下所有类别文件夹，并按名字排序。
    if not class_dirs:  # 如果一个类别文件夹都没有，说明数据集结构不符合“每类一个文件夹”的格式。
        raise RuntimeError(f"No class folders found under: {data_dir}")  # 抛出错误，提醒用户检查数据目录。

    class_to_idx = {path.name: idx for idx, path in enumerate(class_dirs)}  # 建立“类别名 -> 数字标签”的映射，例如 aloe -> 0。
    samples = []  # 创建样本列表，里面每一项会是：音频路径、数字标签、类别名。
    for class_dir in class_dirs:  # 遍历每个类别文件夹，比如 aloe、burger、chips 等。
        files = sorted(  # 找出当前类别文件夹下所有音频文件，并排序保证顺序稳定。
            path  # 这里的 path 是找到的某一个文件路径。
            for path in class_dir.rglob("*")  # 递归扫描当前类别文件夹中的所有文件和子文件夹。
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS  # 只保留真正的文件，并且后缀必须是支持的音频格式。
        )  # 结束当前类别音频文件列表的生成。
        if limit_per_class > 0:  # 如果用户设置了每类最多取多少条，就进入限制逻辑。
            files = files[:limit_per_class]  # 只取排序后的前 N 个音频，适合快速测试脚本是否能跑通。
        label = class_to_idx[class_dir.name]  # 根据类别文件夹名拿到对应的数字标签。
        samples.extend((path, label, class_dir.name) for path in files)  # 把这个类别里的音频全部加入总样本列表。

    if not samples:  # 如果扫描完之后没有任何音频文件，训练也无法开始。
        raise RuntimeError(f"No audio files found under: {data_dir}")  # 抛出错误，提醒用户检查音频文件是否存在。
    return samples, class_to_idx  # 返回样本列表和类别映射，后面的数据划分和训练都会用到。


def group_key(path):  # 根据文件名提取“录音组”编号，避免同一段原始录音切片同时进训练集和验证集。
    parts = path.stem.split("_")  # path.stem 是不带后缀的文件名，比如 fries_9_88；按下划线切成多个部分。
    if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():  # 判断文件名是否像 类别_录音编号_切片编号 这种格式。
        return parts[-2]  # 返回倒数第二段作为录音组编号，例如 fries_9_88 里的 9。
    return path.stem  # 如果文件名不符合预期格式，就退一步用完整文件名当作组编号。


def split_samples(samples, train_ratio, val_ratio, seed):  # 按类别和录音组划分训练集、验证集、测试集。
    rng = random.Random(seed)  # 创建一个独立随机数生成器，避免影响其他 random 操作。
    by_class = defaultdict(lambda: defaultdict(list))  # 创建两层字典：第一层按类别，第二层按录音组，值是样本列表。
    for path, label, class_name in samples:  # 遍历所有样本，逐条放进对应类别和对应录音组。
        by_class[label][group_key(path)].append((path, label, class_name))  # 把同一类、同一录音组的切片放到一起。

    train, val, test = [], [], []  # 准备三个列表，分别存放训练集、验证集和测试集样本。
    for label, groups in by_class.items():  # 对每个类别单独划分，保证每个类别都尽量出现在三个集合里。
        group_items = list(groups.values())  # 把当前类别的所有录音组转换成列表，每个元素是一组音频切片。
        rng.shuffle(group_items)  # 随机打乱录音组顺序，防止按文件名顺序划分造成偏差。

        n_groups = len(group_items)  # 当前类别一共有多少个录音组。
        train_groups = max(1, int(round(n_groups * train_ratio)))  # 按比例计算训练组数量，至少保留 1 组训练。
        val_groups = max(1, int(round(n_groups * val_ratio))) if n_groups >= 3 else 0  # 如果录音组足够多，就至少留 1 组验证。
        if train_groups + val_groups >= n_groups:  # 如果训练组和验证组数量已经占满或超过总组数，就需要调整。
            train_groups = max(1, n_groups - 2)  # 尽量给验证集和测试集各留空间，同时训练集至少 1 组。
            val_groups = 1 if n_groups >= 2 else 0  # 如果至少有 2 组，就给验证集留 1 组。

        selected_train = group_items[:train_groups]  # 取前面一部分录音组作为训练集。
        selected_val = group_items[train_groups : train_groups + val_groups]  # 接着取一部分录音组作为验证集。
        selected_test = group_items[train_groups + val_groups :]  # 剩余录音组作为测试集。

        train.extend(sample for group in selected_train for sample in group)  # 把训练录音组里的所有切片展开加入训练列表。
        val.extend(sample for group in selected_val for sample in group)  # 把验证录音组里的所有切片展开加入验证列表。
        test.extend(sample for group in selected_test for sample in group)  # 把测试录音组里的所有切片展开加入测试列表。

    rng.shuffle(train)  # 再打乱训练集样本顺序，让每个 batch 更随机。
    rng.shuffle(val)  # 打乱验证集样本顺序，虽然验证不训练，但顺序随机也更自然。
    rng.shuffle(test)  # 打乱测试集样本顺序。
    return train, val, test  # 返回划分好的三个数据列表。


class ChewingSoundDataset(Dataset):  # 自定义 PyTorch 数据集类，负责读取音频并变成模型能吃的张量。
    def __init__(self, samples, sample_rate, duration, training=False):  # 初始化数据集，保存样本信息和音频处理参数。
        self.samples = samples  # 保存样本列表，每个样本包含音频路径、标签、类别名。
        self.sample_rate = sample_rate  # 保存目标采样率，例如 16000Hz。
        self.num_samples = int(sample_rate * duration)  # 计算每条音频应有多少采样点，例如 16000 * 4 秒。
        self.training = training  # 保存当前是否是训练模式，训练模式会做随机裁剪和简单增强。

    def __len__(self):  # DataLoader 会调用这个函数来知道数据集有多少条样本。
        return len(self.samples)  # 返回样本数量。

    def __getitem__(self, index):  # DataLoader 会调用这个函数来取第 index 条音频。
        path, label, _ = self.samples[index]  # 从样本列表中取出音频路径和标签，类别名这里暂时不用所以写成 _。
        waveform, source_rate = torchaudio.load(str(path))  # 读取音频文件，waveform 是声音波形，source_rate 是原始采样率。
        waveform = waveform.mean(dim=0, keepdim=True)  # 如果是双声道或多声道，就取平均变成单声道。

        if source_rate != self.sample_rate:  # 如果音频原始采样率和目标采样率不同，就需要重采样。
            waveform = torchaudio.functional.resample(  # 调用 torchaudio 的重采样函数，把声音转换到统一采样率。
                waveform, source_rate, self.sample_rate  # 传入原始波形、原始采样率和目标采样率。
            )  # 结束重采样函数调用。

        waveform = self._crop_or_pad(waveform)  # 把音频裁剪或补零到固定长度，保证所有输入大小一致。
        if self.training:  # 如果当前是训练集，就进行简单数据增强。
            waveform = self._augment(waveform)  # 对音频做随机音量变化和轻微噪声，让模型更抗干扰。

        return waveform, torch.tensor(label, dtype=torch.long)  # 返回音频张量和标签张量，标签必须是 long 类型才能用于交叉熵损失。

    def _crop_or_pad(self, waveform):  # 定义音频裁剪或补齐函数，让每条音频长度一致。
        length = waveform.shape[-1]  # 获取当前音频的采样点数量，也就是音频长度。
        if length > self.num_samples:  # 如果音频比目标长度长，就需要裁剪。
            if self.training:  # 训练时使用随机裁剪，让模型看到同一音频的不同片段。
                start = random.randint(0, length - self.num_samples)  # 随机选择裁剪起点。
            else:  # 验证和测试时不随机，这样每次评估结果更稳定。
                start = (length - self.num_samples) // 2  # 从中间位置裁剪一段固定长度音频。
            waveform = waveform[:, start : start + self.num_samples]  # 按起点和目标长度截取音频。
        elif length < self.num_samples:  # 如果音频比目标长度短，就需要补零。
            pad = self.num_samples - length  # 计算还差多少个采样点。
            waveform = torch.nn.functional.pad(waveform, (0, pad))  # 在音频末尾补 0，使长度达到目标长度。
        return waveform  # 返回长度已经统一的音频波形。

    @staticmethod  # 声明下面的函数不依赖 self，可以直接作为工具函数使用。
    def _augment(waveform):  # 定义训练时使用的简单音频增强函数。
        gain = random.uniform(0.75, 1.25)  # 随机生成音量倍率，让声音有时小一点、有时大一点。
        waveform = waveform * gain  # 把音频波形乘以音量倍率。
        if random.random() < 0.35:  # 有 35% 的概率添加轻微噪声，不是每条都加，避免破坏数据。
            noise = torch.randn_like(waveform) * random.uniform(0.001, 0.006)  # 生成和音频同形状的随机噪声，并控制噪声强度。
            waveform = waveform + noise  # 把噪声加到原始音频上，提高模型对噪声环境的适应能力。
        return waveform.clamp(-1.0, 1.0)  # 把声音数值限制在 -1 到 1 之间，防止增强后幅度过大。


class AudioCNN(nn.Module):  # 定义音频分类神经网络，输入音频波形，输出每个食物类别的分数。
    def __init__(self, num_classes, sample_rate, n_mels):  # 初始化模型结构，需要类别数、采样率和 Mel 频谱维度。
        super().__init__()  # 调用父类 nn.Module 的初始化，这是 PyTorch 模型必须做的一步。
        self.mel = torchaudio.transforms.MelSpectrogram(  # 创建 Mel 频谱转换器，把一维声音波形变成二维“声音图片”。
            sample_rate=sample_rate,  # 告诉转换器音频采样率是多少。
            n_fft=1024,  # FFT 窗口大小，控制每次分析多长的一小段声音。
            hop_length=256,  # 相邻分析窗口之间移动多少采样点，越小时间分辨率越高但计算更多。
            n_mels=n_mels,  # Mel 频率通道数量，默认 96。
            f_min=40,  # 最低分析频率，低于 40Hz 的声音通常对咀嚼分类帮助较小。
            f_max=sample_rate // 2,  # 最高分析频率，不能超过采样率的一半，这是音频采样理论限制。
            power=2.0,  # 使用功率谱，也就是幅度平方，常用于音频特征提取。
        )  # 结束 MelSpectrogram 的创建。
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power")  # 把功率谱转换成分贝尺度，更接近人耳对声音强弱的感知。
        self.features = nn.Sequential(  # 定义卷积特征提取器，把 Mel 频谱逐层提取成高级特征。
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),  # 第一层二维卷积：输入 1 个通道，输出 32 个特征通道。
            nn.BatchNorm2d(32),  # 批归一化，让训练更稳定、更快收敛。
            nn.ReLU(inplace=True),  # ReLU 激活函数，引入非线性，让模型能学习复杂规律。
            nn.MaxPool2d(2),  # 最大池化，把特征图宽高减半，减少计算并保留明显特征。
            nn.Dropout2d(0.10),  # 随机丢弃 10% 的二维特征通道，降低过拟合风险。
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),  # 第二层卷积：从 32 通道提取到 64 通道。
            nn.BatchNorm2d(64),  # 对 64 个通道做批归一化。
            nn.ReLU(inplace=True),  # 再次使用 ReLU 激活。
            nn.MaxPool2d(2),  # 再次池化，进一步压缩时间和频率维度。
            nn.Dropout2d(0.15),  # 丢弃比例提高到 15%，因为特征更多后更容易过拟合。
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),  # 第三层卷积：输出 128 个更抽象的特征通道。
            nn.BatchNorm2d(128),  # 对 128 个通道做批归一化。
            nn.ReLU(inplace=True),  # 使用 ReLU 激活。
            nn.MaxPool2d(2),  # 再池化一次，继续压缩特征图。
            nn.Dropout2d(0.20),  # 丢弃 20% 特征通道，继续减少过拟合。
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),  # 第四层卷积：输出 256 个高级特征通道。
            nn.BatchNorm2d(256),  # 对 256 个通道做批归一化。
            nn.ReLU(inplace=True),  # 使用 ReLU 激活。
            nn.AdaptiveAvgPool2d((1, 1)),  # 自适应平均池化，把每个通道压缩成 1 个数，方便接全连接层。
        )  # 结束卷积特征提取器定义。
        self.classifier = nn.Sequential(  # 定义分类器部分，把 256 个特征变成类别分数。
            nn.Flatten(),  # 把形状从 [batch, 256, 1, 1] 展平成 [batch, 256]。
            nn.Dropout(0.35),  # 分类前再随机丢弃 35% 特征，减少对训练集的死记硬背。
            nn.Linear(256, num_classes),  # 全连接层，输出维度等于类别数，比如 20 类就输出 20 个分数。
        )  # 结束分类器定义。

    def forward(self, waveform):  # 定义模型前向传播：输入音频波形，输出类别 logits。
        x = self.mel(waveform)  # 第一步：把原始波形转换成 Mel 频谱。
        x = self.to_db(x)  # 第二步：把 Mel 功率谱转换成分贝值。
        mean = x.mean(dim=(-2, -1), keepdim=True)  # 计算每条音频频谱的均值，用于标准化。
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)  # 计算标准差，并设置最小值防止除以 0。
        x = (x - mean) / std  # 对频谱做标准化，让不同音量的音频更容易被模型统一处理。
        x = self.features(x)  # 把标准化后的 Mel 频谱送入卷积网络提取特征。
        return self.classifier(x)  # 把特征送入分类器，返回每个类别的原始分数。


def make_loader(samples, args, training):  # 创建 DataLoader，负责按 batch 读取数据。
    dataset = ChewingSoundDataset(  # 先创建自定义音频数据集对象。
        samples=samples,  # 传入当前集合的样本列表，比如训练集样本。
        sample_rate=args.sample_rate,  # 传入目标采样率。
        duration=args.duration,  # 传入每条音频统一的秒数。
        training=training,  # 告诉数据集当前是否训练模式，训练模式会启用增强。
    )  # 结束数据集对象创建。
    return DataLoader(  # 返回 PyTorch DataLoader，它会把 Dataset 包装成批量迭代器。
        dataset,  # 要读取的数据集。
        batch_size=args.batch_size,  # 每个 batch 读取多少条音频。
        shuffle=training,  # 训练集需要打乱，验证和测试集不需要强制打乱。
        num_workers=args.num_workers,  # 读取数据使用多少个子进程。
        pin_memory=torch.cuda.is_available(),  # 如果有 CUDA，固定内存可以加速 CPU 到 GPU 的数据拷贝。
        persistent_workers=args.num_workers > 0,  # 如果启用多进程读取，就让 worker 持续存在，减少反复启动开销。
    )  # 结束 DataLoader 创建。


def class_weights(samples, num_classes):  # 计算类别权重，缓解类别样本数量不均衡的问题。
    labels = torch.tensor([label for _, label, _ in samples], dtype=torch.long)  # 从样本列表里取出所有标签，转成 PyTorch 张量。
    counts = torch.bincount(labels, minlength=num_classes).float()  # 统计每个类别有多少条训练样本。
    return counts.sum() / (counts.clamp_min(1.0) * num_classes)  # 样本少的类别权重大，样本多的类别权重小。


def run_epoch(model, loader, criterion, optimizer, scaler, device, training, use_amp):  # 运行一轮训练或评估，并返回平均 loss 和准确率。
    model.train(training)  # 如果 training=True 就进入训练模式；否则进入评估模式，影响 Dropout 和 BatchNorm 行为。
    total_loss = 0.0  # 累计所有 batch 的 loss 总和，后面会除以样本数求平均。
    total_correct = 0  # 累计预测正确的样本数量。
    total_seen = 0  # 累计已经处理过的样本数量。
    use_amp = use_amp and device.type == "cuda"  # 只有用户开启 --amp 且设备是 CUDA 时，才真的使用混合精度。

    for waveforms, labels in loader:  # 从 DataLoader 中一批一批取出音频和标签。
        waveforms = waveforms.to(device, non_blocking=True)  # 把音频张量移动到 CPU 或 GPU，GPU 训练时这里会放到显卡上。
        labels = labels.to(device, non_blocking=True)  # 把标签也移动到同一个设备上，模型输出和标签必须在同一设备。

        with torch.set_grad_enabled(training):  # 训练时开启梯度计算，验证和测试时关闭梯度以节省显存和计算。
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):  # 如果启用混合精度，部分计算会自动使用更快的低精度。
                logits = model(waveforms)  # 前向传播：把音频送进模型，得到每个类别的预测分数。
                loss = criterion(logits, labels)  # 用损失函数比较预测和真实标签，得到当前 batch 的错误程度。

            if training:  # 只有训练阶段才需要反向传播和更新模型参数。
                optimizer.zero_grad(set_to_none=True)  # 清空上一轮残留的梯度，否则梯度会累加。
                if use_amp:  # 如果开启混合精度，需要用 scaler 防止小梯度下溢。
                    scaler.scale(loss).backward()  # 对放大后的 loss 做反向传播，计算梯度。
                    scaler.step(optimizer)  # 根据缩放后的梯度更新模型参数。
                    scaler.update()  # 更新 scaler 的缩放比例，让混合精度训练更稳定。
                else:  # 如果没有开启混合精度，就走普通 FP32 训练流程。
                    loss.backward()  # 反向传播，计算每个参数应该如何调整。
                    optimizer.step()  # 优化器根据梯度真正更新模型参数。

        batch_size = labels.size(0)  # 获取当前 batch 中样本数量，最后一个 batch 可能小于 batch_size。
        total_loss += loss.item() * batch_size  # 累计 loss，总 loss 用 batch 平均 loss 乘以 batch 样本数。
        total_correct += (logits.argmax(dim=1) == labels).sum().item()  # 统计当前 batch 预测正确的数量。
        total_seen += batch_size  # 累计已经处理的样本数量。

    if total_seen == 0:  # 如果这个 loader 没有样本，避免后面除以 0 报错。
        return 0.0, 0.0  # 空数据集时返回 0 loss 和 0 准确率。
    return total_loss / total_seen, total_correct / total_seen  # 返回平均 loss 和准确率。


def save_json(path, data):  # 定义保存 JSON 文件的工具函数。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保 JSON 文件所在目录存在，不存在就自动创建。
    with path.open("w", encoding="utf-8") as handle:  # 以 UTF-8 编码打开文件，避免中文或特殊字符乱码。
        json.dump(data, handle, indent=2, ensure_ascii=False)  # 把 Python 数据写成格式漂亮的 JSON，ensure_ascii=False 保留原字符。


def main():  # 主函数，脚本真正的执行流程从这里开始组织。
    args = parse_args()  # 读取命令行参数，比如数据目录、训练轮数、batch 大小等。
    set_seed(args.seed)  # 设置随机种子，让数据划分和训练随机性尽量可复现。

    if torch.cuda.is_available():  # 判断当前电脑是否能使用 NVIDIA CUDA GPU。
        torch.backends.cudnn.benchmark = True  # 让 cuDNN 自动寻找更快的卷积算法，固定输入尺寸时通常能加速。
        device = torch.device("cuda")  # 如果有 CUDA，就把训练设备设为 GPU。
    else:  # 如果没有可用 GPU，就只能用 CPU 训练。
        device = torch.device("cpu")  # 把训练设备设为 CPU。
        print("CUDA is not available. Training will run on CPU.")  # 打印提醒，让用户知道当前没有用上 GPU。

    samples, class_to_idx = list_audio_files(args.data_dir, args.limit_per_class)  # 扫描音频文件，并生成类别名到标签编号的映射。
    train_samples, val_samples, test_samples = split_samples(  # 把所有样本划分为训练集、验证集和测试集。
        samples, args.train_ratio, args.val_ratio, args.seed  # 传入样本列表、训练比例、验证比例和随机种子。
    )  # 结束样本划分函数调用。
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}  # 生成反向映射：数字标签 -> 类别名，便于保存和查看。

    args.output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录，模型和 JSON 指标会保存在这里。
    save_json(args.output_dir / "class_to_idx.json", class_to_idx)  # 保存类别名到数字标签的映射，预测时需要知道编号对应哪个食物。
    save_json(  # 保存数据划分摘要，方便确认训练集、验证集、测试集各有多少条。
        args.output_dir / "split_summary.json",  # 摘要文件保存路径。
        {  # 下面这个字典就是要写入 JSON 的内容。
            "train": len(train_samples),  # 训练集样本数量。
            "val": len(val_samples),  # 验证集样本数量。
            "test": len(test_samples),  # 测试集样本数量。
            "classes": idx_to_class,  # 数字标签到类别名的映射。
        },  # 数据划分摘要字典结束。
    )  # 结束保存 split_summary.json。

    train_loader = make_loader(train_samples, args, training=True)  # 创建训练集 DataLoader，训练模式会启用数据增强和打乱。
    val_loader = make_loader(val_samples, args, training=False)  # 创建验证集 DataLoader，不做训练增强。
    test_loader = make_loader(test_samples, args, training=False)  # 创建测试集 DataLoader，不做训练增强。

    model = AudioCNN(  # 创建音频 CNN 分类模型。
        num_classes=len(class_to_idx),  # 输出类别数等于数据集中类别文件夹数量，这里是 20 类。
        sample_rate=args.sample_rate,  # 传入采样率，让模型内部 Mel 频谱转换器知道音频频率范围。
        n_mels=args.n_mels,  # 传入 Mel 频谱通道数量。
    ).to(device)  # 把模型移动到 GPU 或 CPU 上，和数据保持同一设备。

    criterion = nn.CrossEntropyLoss(  # 创建交叉熵损失函数，多分类任务最常用。
        weight=class_weights(train_samples, len(class_to_idx)).to(device)  # 加入类别权重，样本少的类别会被更重视。
    )  # 结束损失函数创建。
    optimizer = torch.optim.AdamW(  # 创建 AdamW 优化器，用来根据梯度更新模型参数。
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay  # 传入模型参数、学习率和权重衰减。
    )  # 结束优化器创建。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(  # 创建余弦退火学习率调度器，让学习率随训练逐渐变化。
        optimizer, T_max=args.epochs  # 告诉调度器控制哪个优化器，以及总训练轮数。
    )  # 结束学习率调度器创建。
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")  # 创建混合精度缩放器，只有 --amp 且 CUDA 时启用。

    print(f"Device: {device}")  # 打印当前使用的训练设备，cuda 表示 GPU，cpu 表示 CPU。
    if device.type == "cuda":  # 如果当前设备是 GPU，就额外打印显卡名称。
        print(f"GPU: {torch.cuda.get_device_name(0)}")  # 打印第 0 块 GPU 的名称。
    print(f"Classes: {len(class_to_idx)}")  # 打印类别数量。
    print(  # 打印训练集、验证集、测试集样本数量。
        f"Samples: train={len(train_samples)}, val={len(val_samples)}, "  # 第一段字符串，显示训练集和验证集数量。
        f"test={len(test_samples)}"  # 第二段字符串，显示测试集数量。
    )  # 结束打印样本数量。

    history = []  # 用列表保存每一轮的训练 loss、训练准确率、验证 loss、验证准确率等信息。
    best_val_acc = 0.0  # 记录目前为止最好的验证集准确率。
    best_path = args.output_dir / "best_model.pt"  # 最优模型的保存路径。

    for epoch in range(1, args.epochs + 1):  # 从第 1 轮训练到第 epochs 轮，默认共 50 轮。
        train_loss, train_acc = run_epoch(  # 运行一轮训练，并得到训练 loss 和训练准确率。
            model,  # 要训练的模型。
            train_loader,  # 训练集 DataLoader。
            criterion,  # 损失函数。
            optimizer,  # 优化器，训练时用于更新参数。
            scaler,  # 混合精度缩放器，未开启 amp 时不会真正使用。
            device,  # 当前设备，cuda 或 cpu。
            training=True,  # 告诉 run_epoch 当前是训练模式。
            use_amp=args.amp,  # 是否使用混合精度，由命令行 --amp 控制。
        )  # 结束训练轮函数调用。
        val_loss, val_acc = run_epoch(  # 运行一轮验证，并得到验证 loss 和验证准确率。
            model,  # 要评估的模型。
            val_loader,  # 验证集 DataLoader。
            criterion,  # 同一个损失函数，用来计算验证 loss。
            optimizer,  # 优化器参数在验证时不会更新，但函数签名统一所以传入。
            scaler,  # 混合精度缩放器，验证时不会做反向传播。
            device,  # 当前设备，cuda 或 cpu。
            training=False,  # 告诉 run_epoch 当前是验证模式。
            use_amp=args.amp,  # 是否使用混合精度。
        )  # 结束验证轮函数调用。
        scheduler.step()  # 每轮结束后更新学习率。

        row = {  # 创建一个字典，记录当前这一轮的训练和验证结果。
            "epoch": epoch,  # 当前训练轮数。
            "train_loss": train_loss,  # 当前轮训练集平均 loss。
            "train_acc": train_acc,  # 当前轮训练集准确率。
            "val_loss": val_loss,  # 当前轮验证集平均 loss。
            "val_acc": val_acc,  # 当前轮验证集准确率。
            "lr": scheduler.get_last_lr()[0],  # 当前学习率，方便之后分析训练过程。
        }  # 当前轮历史记录字典结束。
        history.append(row)  # 把当前轮结果加入训练历史列表。
        print(  # 打印当前轮训练结果，让用户能实时看到训练进展。
            f"Epoch {epoch:03d}/{args.epochs} | "  # 打印当前是第几轮，例如 001/050。
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "  # 打印训练 loss 和训练准确率，保留 4 位小数。
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"  # 打印验证 loss 和验证准确率，保留 4 位小数。
        )  # 结束当前轮日志打印。

        if val_acc >= best_val_acc:  # 如果当前验证准确率不低于历史最好结果，就保存当前模型。
            best_val_acc = val_acc  # 更新历史最佳验证准确率。
            torch.save(  # 使用 PyTorch 保存模型 checkpoint。
                {  # checkpoint 是一个字典，里面保存模型参数和训练配置。
                    "epoch": epoch,  # 保存当前是第几轮得到的最好模型。
                    "model_state": model.state_dict(),  # 保存模型参数，这是之后加载模型最重要的内容。
                    "class_to_idx": class_to_idx,  # 保存类别映射，预测时需要把输出编号转回类别名。
                    "sample_rate": args.sample_rate,  # 保存训练时使用的采样率，预测时应保持一致。
                    "duration": args.duration,  # 保存训练时使用的音频长度，预测时应保持一致。
                    "n_mels": args.n_mels,  # 保存 Mel 频谱维度，重建模型时要用。
                    "val_acc": val_acc,  # 保存当前模型对应的验证准确率。
                    "args": vars(args),  # 保存所有命令行参数，方便之后复现实验。
                },  # checkpoint 字典结束。
                best_path,  # 保存到 best_model.pt。
            )  # 结束最优模型保存。

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)  # 训练完成后加载验证集上表现最好的模型。
    model.load_state_dict(checkpoint["model_state"])  # 把最优模型参数加载回当前模型。
    test_loss, test_acc = run_epoch(  # 使用最优模型在测试集上做最终评估。
        model,  # 已加载最优权重的模型。
        test_loader,  # 测试集 DataLoader。
        criterion,  # 损失函数，用来计算测试 loss。
        optimizer,  # 优化器，测试时不会更新参数。
        scaler,  # 混合精度缩放器，测试时不会做反向传播。
        device,  # 当前设备，cuda 或 cpu。
        training=False,  # 测试阶段不是训练模式。
        use_amp=args.amp,  # 是否使用混合精度。
    )  # 结束测试评估。

    torch.save(  # 保存 last_model.pt，这里保存的是加载过最优权重后的模型状态。
        {  # 要保存的 checkpoint 字典。
            "epoch": args.epochs,  # 记录总训练轮数。
            "model_state": model.state_dict(),  # 保存模型参数。
            "class_to_idx": class_to_idx,  # 保存类别映射。
            "sample_rate": args.sample_rate,  # 保存采样率。
            "duration": args.duration,  # 保存音频长度。
            "n_mels": args.n_mels,  # 保存 Mel 频谱维度。
            "test_acc": test_acc,  # 保存测试集准确率。
            "args": vars(args),  # 保存训练参数。
        },  # checkpoint 字典结束。
        args.output_dir / "last_model.pt",  # 保存路径。
    )  # 结束 last_model.pt 保存。
    save_json(args.output_dir / "history.json", history)  # 保存完整训练历史，每一轮的指标都会在这个文件里。
    save_json(  # 保存最终核心指标，方便快速查看本次训练效果。
        args.output_dir / "metrics.json",  # 指标文件保存路径。
        {  # 指标字典开始。
            "best_val_acc": best_val_acc,  # 最好的验证集准确率。
            "test_loss": test_loss,  # 测试集 loss。
            "test_acc": test_acc,  # 测试集准确率。
            "best_model": str(best_path),  # 最优模型文件路径。
        },  # 指标字典结束。
    )  # 结束保存 metrics.json。

    print(f"Best validation accuracy: {best_val_acc:.4f}")  # 打印最佳验证准确率。
    print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")  # 打印测试集 loss 和测试准确率。
    print(f"Saved best model to: {best_path}")  # 打印最优模型保存位置。


if __name__ == "__main__":  # Python 脚本入口：只有直接运行这个文件时，下面的 main() 才会执行。
    main()  # 调用主函数，开始完整训练流程。
