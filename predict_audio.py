"""使用训练好的模型预测单个音频文件。

运行示例：
python predict_audio.py path/to/audio.wav --model runs/chewing_audio/best_model.pt --top-k 5
"""

import argparse  # 用于读取命令行参数。
from pathlib import Path  # 用 pathlib 处理文件路径。

import torch  # PyTorch：加载模型、张量计算、GPU 推理。
import torchaudio  # torchaudio：读取音频和重采样。

from train_chewing_audio import AudioCNN  # 复用训练脚本中的模型结构，保证训练和预测结构一致。


def parse_args():
    """解析预测脚本的命令行参数。"""
    parser = argparse.ArgumentParser(description="Predict one chewing sound audio file.")
    parser.add_argument("audio", type=Path, help="Path to one audio file, such as a .wav file.")  # 必填：要预测的音频路径。
    parser.add_argument("--model", type=Path, default=Path("runs/chewing_audio/best_model.pt"))  # 模型 checkpoint 路径。
    parser.add_argument("--top-k", type=int, default=5)  # 输出概率最高的前 K 个类别。
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")  # 推理设备。
    return parser.parse_args()


def choose_device(device_arg):
    """根据用户参数选择 CPU 或 CUDA。"""
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("You requested CUDA, but CUDA is not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    # auto 模式：有 GPU 就用 GPU，没有就用 CPU。
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model_path, device):
    """加载训练保存的 checkpoint。"""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return torch.load(model_path, map_location=device, weights_only=False)


def checkpoint_arg(checkpoint, name, default):
    """从 checkpoint 中读取参数。

    新版 checkpoint 会把部分参数放在顶层，也会把完整命令行参数放在 args 字典里。
    这个函数用于兼容新旧 checkpoint。
    """
    saved_args = checkpoint.get("args", {})
    return checkpoint.get(name, saved_args.get(name, default))


def load_model(checkpoint, device):
    """根据 checkpoint 重建模型并加载权重。"""
    class_to_idx = checkpoint["class_to_idx"]  # 类别名到数字编号的映射。
    sample_rate = checkpoint["sample_rate"]  # 训练时使用的采样率。
    n_mels = checkpoint["n_mels"]  # 训练时使用的 Mel 频谱通道数。
    model_size = checkpoint_arg(checkpoint, "model_size", "legacy")  # 兼容旧模型，旧 checkpoint 默认 legacy。
    dropout = checkpoint_arg(checkpoint, "dropout", 0.35)

    model = AudioCNN(
        num_classes=len(class_to_idx),
        sample_rate=sample_rate,
        n_mels=n_mels,
        model_size=model_size,
        dropout=dropout,
        specaugment=False,  # 推理时必须关闭 SpecAugment，保证预测结果稳定。
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # 进入评估模式，关闭 Dropout、BatchNorm 更新等训练行为。
    return model


def crop_or_pad(waveform, target_samples):
    """把音频裁剪或补零到模型要求的固定长度。"""
    length = waveform.shape[-1]
    if length > target_samples:
        start = (length - target_samples) // 2  # 推理时使用中心裁剪，保证稳定。
        waveform = waveform[:, start : start + target_samples]
    elif length < target_samples:
        pad = target_samples - length
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    return waveform


def load_audio(audio_path, sample_rate, duration, device):
    """读取并预处理待预测音频。"""
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform, source_rate = torchaudio.load(str(audio_path))  # 读取音频。
    waveform = waveform.mean(dim=0, keepdim=True)  # 多声道转单声道。

    # 如果音频采样率和训练采样率不同，需要重采样。
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)

    # 幅度归一化，和训练脚本保持一致。
    peak = waveform.abs().max().clamp_min(1e-6)
    waveform = waveform / peak

    target_samples = int(sample_rate * duration)
    waveform = crop_or_pad(waveform, target_samples)
    waveform = waveform.unsqueeze(0)  # 增加 batch 维度：[1, 1, samples]。
    return waveform.to(device)


def predict(model, waveform, top_k):
    """执行模型推理，返回 top-k 概率和类别编号。"""
    with torch.inference_mode():  # 推理阶段不需要梯度，节省显存和计算。
        logits = model(waveform)
        probabilities = torch.softmax(logits, dim=1)  # 转成概率。
        top_probs, top_indices = probabilities.topk(top_k, dim=1)
    return top_probs[0].cpu(), top_indices[0].cpu()


def main():
    """预测脚本主流程。"""
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.model, device)
    model = load_model(checkpoint, device)

    # 把“类别名 -> 编号”反转为“编号 -> 类别名”，方便打印结果。
    idx_to_class = {idx: name for name, idx in checkpoint["class_to_idx"].items()}

    waveform = load_audio(args.audio, checkpoint["sample_rate"], checkpoint["duration"], device)
    top_k = min(args.top_k, len(idx_to_class))
    top_probs, top_indices = predict(model, waveform, top_k)

    print(f"Device: {device}")
    print(f"Audio: {args.audio}")
    print("Top predictions:")
    for rank, (probability, class_index) in enumerate(zip(top_probs, top_indices), start=1):
        class_name = idx_to_class[int(class_index)]
        print(f"{rank}. {class_name}: {probability.item() * 100:.2f}%")


if __name__ == "__main__":
    main()
