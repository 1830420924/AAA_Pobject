import argparse  # 导入 argparse：用来读取命令行参数，比如要预测哪个音频、模型路径在哪里。
from pathlib import Path  # 导入 Path：用更清晰的方式处理文件路径。

import torch  # 导入 PyTorch：用于加载模型、张量计算、GPU 推理。
import torchaudio  # 导入 torchaudio：用于读取音频文件和重采样。

from train_chewing_audio import AudioCNN  # 从训练脚本中导入同一个模型结构，保证推理模型和训练模型完全一致。


def parse_args():  # 定义命令行参数解析函数。
    parser = argparse.ArgumentParser(description="Predict one chewing sound audio file.")  # 创建参数解析器，并写一句脚本说明。
    parser.add_argument("audio", type=Path, help="Path to one audio file, such as a .wav file.")  # 必填参数：需要预测的单个音频文件路径。
    parser.add_argument("--model", type=Path, default=Path("runs/chewing_audio/best_model.pt"))  # 模型路径，默认读取训练时保存的最优模型。
    parser.add_argument("--top-k", type=int, default=5)  # 输出概率最高的前 K 个类别，默认显示前 5 个。
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")  # 选择推理设备，auto 表示有 GPU 就用 GPU。
    return parser.parse_args()  # 解析命令行输入，并返回参数对象。


def choose_device(device_arg):  # 根据命令行参数选择 CPU 或 GPU。
    if device_arg == "cuda":  # 如果用户明确指定使用 cuda。
        if not torch.cuda.is_available():  # 但当前环境没有可用 CUDA。
            raise RuntimeError("You requested CUDA, but CUDA is not available.")  # 抛出错误，提醒用户设备不可用。
        return torch.device("cuda")  # 返回 CUDA 设备对象。
    if device_arg == "cpu":  # 如果用户明确指定使用 CPU。
        return torch.device("cpu")  # 返回 CPU 设备对象。
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # auto 模式：有 GPU 用 GPU，没有 GPU 用 CPU。


def load_checkpoint(model_path, device):  # 加载训练保存的模型 checkpoint。
    if not model_path.exists():  # 如果模型文件不存在。
        raise FileNotFoundError(f"Model file not found: {model_path}")  # 抛出清楚的文件不存在错误。
    return torch.load(model_path, map_location=device, weights_only=False)  # 加载 checkpoint，并映射到当前推理设备。


def load_model(checkpoint, device):  # 根据 checkpoint 重建模型结构并加载参数。
    class_to_idx = checkpoint["class_to_idx"]  # 读取类别名到数字标签的映射，例如 chips -> 5。
    sample_rate = checkpoint["sample_rate"]  # 读取训练时使用的采样率，预测时必须保持一致。
    n_mels = checkpoint["n_mels"]  # 读取训练时使用的 Mel 频谱维度，重建模型时必须保持一致。
    model = AudioCNN(  # 创建和训练时一样的 AudioCNN 模型。
        num_classes=len(class_to_idx),  # 类别数量等于 class_to_idx 中的类别数。
        sample_rate=sample_rate,  # 传入训练时使用的采样率。
        n_mels=n_mels,  # 传入训练时使用的 Mel 频谱维度。
    ).to(device)  # 把模型移动到 GPU 或 CPU。
    model.load_state_dict(checkpoint["model_state"])  # 加载训练好的模型参数。
    model.eval()  # 切换到评估模式，关闭 Dropout，并让 BatchNorm 使用稳定统计。
    return model  # 返回已经可以用于预测的模型。


def crop_or_pad(waveform, target_samples):  # 把音频裁剪或补零到固定长度。
    length = waveform.shape[-1]  # 获取当前音频采样点数量。
    if length > target_samples:  # 如果音频比模型需要的长度更长。
        start = (length - target_samples) // 2  # 推理时选择中间片段，结果更稳定。
        waveform = waveform[:, start : start + target_samples]  # 从中间裁剪出固定长度。
    elif length < target_samples:  # 如果音频比模型需要的长度更短。
        pad = target_samples - length  # 计算需要补多少个采样点。
        waveform = torch.nn.functional.pad(waveform, (0, pad))  # 在音频末尾补 0，使长度一致。
    return waveform  # 返回长度已经统一的音频。


def load_audio(audio_path, sample_rate, duration, device):  # 读取并预处理单个音频文件。
    if not audio_path.exists():  # 如果音频路径不存在。
        raise FileNotFoundError(f"Audio file not found: {audio_path}")  # 抛出清楚的文件不存在错误。
    waveform, source_rate = torchaudio.load(str(audio_path))  # 读取音频，得到波形和原始采样率。
    waveform = waveform.mean(dim=0, keepdim=True)  # 如果是双声道或多声道，就平均成单声道。
    if source_rate != sample_rate:  # 如果音频原始采样率和训练采样率不同。
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)  # 重采样到训练时的采样率。
    target_samples = int(sample_rate * duration)  # 计算模型需要的采样点数量，例如 16000Hz * 4 秒。
    waveform = crop_or_pad(waveform, target_samples)  # 把音频变成固定长度。
    waveform = waveform.unsqueeze(0)  # 增加 batch 维度，从 [1, samples] 变成 [1, 1, samples]。
    return waveform.to(device)  # 把音频移动到 GPU 或 CPU，并返回。


def predict(model, waveform, top_k):  # 用模型预测音频类别，并取概率最高的前 K 个结果。
    with torch.inference_mode():  # 推理时不需要梯度，关闭梯度可以节省显存和加快速度。
        logits = model(waveform)  # 前向传播，得到每个类别的原始分数。
        probabilities = torch.softmax(logits, dim=1)  # 把原始分数转换成概率，总和为 1。
        top_probs, top_indices = probabilities.topk(top_k, dim=1)  # 取概率最高的 top_k 个类别。
    return top_probs[0].cpu(), top_indices[0].cpu()  # 返回 CPU 上的一维概率和类别编号，方便打印。


def main():  # 主函数，组织完整预测流程。
    args = parse_args()  # 读取命令行参数。
    device = choose_device(args.device)  # 根据参数选择推理设备。
    checkpoint = load_checkpoint(args.model, device)  # 加载模型 checkpoint。
    model = load_model(checkpoint, device)  # 重建模型并加载训练好的权重。
    idx_to_class = {idx: name for name, idx in checkpoint["class_to_idx"].items()}  # 把“类别名 -> 编号”反转成“编号 -> 类别名”。
    waveform = load_audio(  # 读取并预处理用户指定的音频文件。
        args.audio,  # 音频路径。
        checkpoint["sample_rate"],  # 使用训练时保存的采样率。
        checkpoint["duration"],  # 使用训练时保存的音频长度。
        device,  # 把音频放到当前推理设备。
    )  # 结束音频预处理。
    top_k = min(args.top_k, len(idx_to_class))  # 防止 top_k 大于类别数量。
    top_probs, top_indices = predict(model, waveform, top_k)  # 执行预测，得到概率最高的若干类别。

    print(f"Device: {device}")  # 打印当前使用 CPU 还是 GPU。
    print(f"Audio: {args.audio}")  # 打印正在预测的音频路径。
    print("Top predictions:")  # 打印标题。
    for rank, (probability, class_index) in enumerate(zip(top_probs, top_indices), start=1):  # 逐个打印 top-k 结果。
        class_name = idx_to_class[int(class_index)]  # 根据类别编号找到类别名称。
        print(f"{rank}. {class_name}: {probability.item() * 100:.2f}%")  # 打印排名、类别名和百分比概率。


if __name__ == "__main__":  # 脚本入口：只有直接运行 predict_audio.py 时才会执行 main。
    main()  # 调用主函数，开始预测。
