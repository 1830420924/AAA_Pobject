import argparse  # 导入 argparse：用来读取命令行参数，比如模型路径、数据路径、输出目录。
import csv  # 导入 csv：用来把混淆矩阵保存成 .csv 表格文件。
from pathlib import Path  # 导入 Path：用更清晰、更安全的方式处理文件和文件夹路径。

import matplotlib.pyplot as plt  # 导入 matplotlib：用来把混淆矩阵画成图片。
import numpy as np  # 导入 numpy：用来处理二维数组，混淆矩阵本质上就是一个二维表。
import torch  # 导入 PyTorch：用于加载模型、GPU 推理和张量计算。
from torch.utils.data import DataLoader  # 导入 DataLoader：用于按 batch 读取验证集或测试集。

from train_chewing_audio import AudioCNN  # 导入训练时使用的模型结构，保证评估时模型一致。
from train_chewing_audio import ChewingSoundDataset  # 导入训练脚本中的数据集类，保证音频预处理一致。
from train_chewing_audio import list_audio_files  # 导入扫描音频文件的函数，复用训练时的类别读取逻辑。
from train_chewing_audio import split_samples  # 导入数据划分函数，保证训练/验证/测试划分规则一致。


def parse_args():  # 定义命令行参数解析函数。
    parser = argparse.ArgumentParser(description="Evaluate a chewing sound classifier and save confusion matrix.")  # 创建参数解析器。
    parser.add_argument("--data-dir", type=Path, default=Path("archive/clips_rd"))  # 数据集目录，默认使用训练脚本同一个目录。
    parser.add_argument("--model", type=Path, default=Path("runs/chewing_audio/best_model.pt"))  # 要评估的模型文件，默认是最优模型。
    parser.add_argument("--output-dir", type=Path, default=Path("runs/chewing_audio/evaluation"))  # 混淆矩阵和评估结果保存目录。
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")  # 选择评估哪个集合，默认评估测试集。
    parser.add_argument("--batch-size", type=int, default=64)  # 评估时每个 batch 的音频数量，通常可以比训练 batch 稍大。
    parser.add_argument("--num-workers", type=int, default=0)  # DataLoader 子进程数量，Windows 默认 0 最稳。
    parser.add_argument("--seed", type=int, default=None)  # 随机种子，默认从 checkpoint 中读取训练时用的 seed。
    parser.add_argument("--train-ratio", type=float, default=None)  # 训练集比例，默认从 checkpoint 中读取训练时配置。
    parser.add_argument("--val-ratio", type=float, default=None)  # 验证集比例，默认从 checkpoint 中读取训练时配置。
    parser.add_argument("--normalize", action="store_true")  # 如果加上这个参数，额外保存按行归一化后的混淆矩阵图片。
    return parser.parse_args()  # 解析命令行参数并返回。


def choose_device():  # 自动选择评估设备。
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 有 CUDA 就用 GPU，否则用 CPU。


def load_checkpoint(model_path, device):  # 加载训练好的模型 checkpoint。
    if not model_path.exists():  # 如果模型文件不存在。
        raise FileNotFoundError(f"Model file not found: {model_path}")  # 抛出清楚的错误信息。
    return torch.load(model_path, map_location=device, weights_only=False)  # 加载 checkpoint，并映射到当前设备。


def checkpoint_arg(checkpoint, name, default):  # 从 checkpoint 中读取训练参数，如果没有就使用默认值。
    saved_args = checkpoint.get("args", {})  # checkpoint 里通常有 args 字典，里面保存训练时命令行参数。
    return saved_args.get(name, default)  # 优先返回保存的参数；如果没有这个键，就返回默认值。


def resolve_split_config(args, checkpoint):  # 确定数据划分参数，尽量和训练时保持一致。
    seed = args.seed if args.seed is not None else checkpoint_arg(checkpoint, "seed", 42)  # 用户没指定 seed 时，使用训练时保存的 seed。
    train_ratio = args.train_ratio if args.train_ratio is not None else checkpoint_arg(checkpoint, "train_ratio", 0.8)  # 用户没指定训练比例时，用训练时配置。
    val_ratio = args.val_ratio if args.val_ratio is not None else checkpoint_arg(checkpoint, "val_ratio", 0.1)  # 用户没指定验证比例时，用训练时配置。
    return seed, train_ratio, val_ratio  # 返回最终使用的划分参数。


def check_class_mapping(current_mapping, checkpoint_mapping):  # 检查当前数据集类别映射是否和训练时一致。
    if current_mapping != checkpoint_mapping:  # 如果类别名到编号的映射不同，模型输出编号就可能对应错类别。
        raise RuntimeError(  # 直接报错比给出错误评估结果更安全。
            "Class mapping mismatch between dataset and checkpoint. "
            "Please evaluate with the same dataset folder used for training."
        )  # 结束错误信息。


def select_samples(split_name, train_samples, val_samples, test_samples):  # 根据用户选择返回对应集合。
    if split_name == "train":  # 如果用户选择训练集。
        return train_samples  # 返回训练集样本。
    if split_name == "val":  # 如果用户选择验证集。
        return val_samples  # 返回验证集样本。
    return test_samples  # 默认返回测试集样本。


def load_model(checkpoint, device):  # 根据 checkpoint 重建模型并加载权重。
    class_to_idx = checkpoint["class_to_idx"]  # 读取类别映射。
    model = AudioCNN(  # 创建和训练时相同的模型。
        num_classes=len(class_to_idx),  # 类别数量。
        sample_rate=checkpoint["sample_rate"],  # 训练时的采样率。
        n_mels=checkpoint["n_mels"],  # 训练时的 Mel 频谱通道数。
    ).to(device)  # 把模型移动到 GPU 或 CPU。
    model.load_state_dict(checkpoint["model_state"])  # 加载训练好的模型参数。
    model.eval()  # 切换到评估模式，关闭 Dropout。
    return model  # 返回可评估的模型。


def make_eval_loader(samples, checkpoint, batch_size, num_workers):  # 创建评估用 DataLoader。
    dataset = ChewingSoundDataset(  # 使用训练脚本里的数据集类，保证重采样、裁剪、补零逻辑一致。
        samples=samples,  # 要评估的样本列表。
        sample_rate=checkpoint["sample_rate"],  # 使用训练时采样率。
        duration=checkpoint["duration"],  # 使用训练时音频长度。
        training=False,  # 评估时不启用随机裁剪和数据增强。
    )  # 数据集对象创建完成。
    return DataLoader(  # 创建 DataLoader。
        dataset,  # 传入数据集对象。
        batch_size=batch_size,  # 每个 batch 的样本数量。
        shuffle=False,  # 评估时不需要打乱顺序。
        num_workers=num_workers,  # 数据读取子进程数量。
        pin_memory=torch.cuda.is_available(),  # 有 GPU 时固定内存，加快 CPU 到 GPU 的拷贝。
        persistent_workers=num_workers > 0,  # 如果有 worker，就保持 worker 存活以减少开销。
    )  # 返回 DataLoader。


def evaluate(model, loader, num_classes, device):  # 在指定数据集上评估模型，并生成混淆矩阵。
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)  # 创建 num_classes x num_classes 的全 0 混淆矩阵。
    total_correct = 0  # 记录预测正确的样本数量。
    total_seen = 0  # 记录评估过的总样本数量。

    with torch.inference_mode():  # 推理评估不需要梯度，关闭梯度能节省显存和计算。
        for waveforms, labels in loader:  # 按 batch 遍历数据。
            waveforms = waveforms.to(device, non_blocking=True)  # 把音频移动到 GPU 或 CPU。
            labels = labels.to(device, non_blocking=True)  # 把标签移动到同一个设备。
            logits = model(waveforms)  # 前向传播，得到每个类别的预测分数。
            predictions = logits.argmax(dim=1)  # 取分数最高的类别作为最终预测类别。
            total_correct += (predictions == labels).sum().item()  # 累计当前 batch 预测正确数量。
            total_seen += labels.numel()  # 累计当前 batch 样本数量。

            labels_cpu = labels.cpu().numpy()  # 把真实标签转到 CPU，并变成 numpy 数组。
            predictions_cpu = predictions.cpu().numpy()  # 把预测标签转到 CPU，并变成 numpy 数组。
            for true_label, predicted_label in zip(labels_cpu, predictions_cpu):  # 逐个样本更新混淆矩阵。
                confusion[true_label, predicted_label] += 1  # 行表示真实类别，列表示预测类别，对应格子加 1。

    accuracy = total_correct / total_seen if total_seen else 0.0  # 计算整体准确率，避免 total_seen 为 0 时除以 0。
    return confusion, accuracy, total_seen  # 返回混淆矩阵、准确率和样本数量。


def save_confusion_csv(path, confusion, class_names):  # 把混淆矩阵保存成 CSV 文件。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在。
    with path.open("w", newline="", encoding="utf-8-sig") as handle:  # 用 utf-8-sig 编码，Excel 打开中文或英文都更稳。
        writer = csv.writer(handle)  # 创建 CSV 写入器。
        writer.writerow(["true\\pred"] + class_names)  # 第一行写表头，列名是预测类别。
        for class_name, row in zip(class_names, confusion):  # 逐行写入，每一行对应一个真实类别。
            writer.writerow([class_name] + row.tolist())  # 第一列是真实类别名，后面是被预测成各类别的数量。


def normalized_confusion(confusion):  # 计算按行归一化的混淆矩阵。
    row_sums = confusion.sum(axis=1, keepdims=True)  # 计算每个真实类别的样本总数。
    row_sums[row_sums == 0] = 1  # 防止某一类样本数为 0 时除以 0。
    return confusion / row_sums  # 每一行除以该真实类别总数，得到比例。


def save_confusion_plot(path, confusion, class_names, title, normalized=False):  # 把混淆矩阵画成图片并保存。
    matrix = normalized_confusion(confusion) if normalized else confusion  # 根据参数决定画数量矩阵还是比例矩阵。
    fig_size = max(10, len(class_names) * 0.55)  # 根据类别数量自动放大图片尺寸，防止文字挤在一起。
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))  # 创建画布和坐标轴。
    image = ax.imshow(matrix, cmap="Blues")  # 用蓝色渐变显示矩阵，颜色越深表示数值越大。
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)  # 添加右侧颜色条，帮助理解颜色代表的数值大小。

    ax.set_title(title)  # 设置图片标题。
    ax.set_xlabel("Predicted label")  # 设置横轴标题：预测类别。
    ax.set_ylabel("True label")  # 设置纵轴标题：真实类别。
    ax.set_xticks(np.arange(len(class_names)))  # 设置横轴刻度位置。
    ax.set_yticks(np.arange(len(class_names)))  # 设置纵轴刻度位置。
    ax.set_xticklabels(class_names, rotation=45, ha="right")  # 设置横轴类别名，并旋转 45 度防止重叠。
    ax.set_yticklabels(class_names)  # 设置纵轴类别名。

    threshold = matrix.max() * 0.6 if matrix.size else 0  # 计算文字颜色阈值，深色背景上用白字更清楚。
    for row_index in range(matrix.shape[0]):  # 遍历矩阵每一行。
        for col_index in range(matrix.shape[1]):  # 遍历矩阵每一列。
            value = matrix[row_index, col_index]  # 取当前格子的值。
            text = f"{value:.2f}" if normalized else str(int(value))  # 归一化图显示小数，数量图显示整数。
            color = "white" if value > threshold else "black"  # 根据背景深浅选择白字或黑字。
            ax.text(col_index, row_index, text, ha="center", va="center", color=color, fontsize=7)  # 在格子中央写数值。

    fig.tight_layout()  # 自动调整布局，减少文字被裁掉的可能。
    path.parent.mkdir(parents=True, exist_ok=True)  # 确保保存目录存在。
    fig.savefig(path, dpi=200)  # 保存图片，dpi=200 让图片更清晰。
    plt.close(fig)  # 关闭画布，释放内存。


def main():  # 主函数，组织完整评估流程。
    args = parse_args()  # 读取命令行参数。
    device = choose_device()  # 自动选择 GPU 或 CPU。
    checkpoint = load_checkpoint(args.model, device)  # 加载训练好的模型 checkpoint。
    seed, train_ratio, val_ratio = resolve_split_config(args, checkpoint)  # 获取和训练时一致的数据划分参数。

    samples, class_to_idx = list_audio_files(args.data_dir)  # 扫描数据目录，得到所有音频样本和类别映射。
    check_class_mapping(class_to_idx, checkpoint["class_to_idx"])  # 检查当前类别编号是否和训练时一致。
    train_samples, val_samples, test_samples = split_samples(samples, train_ratio, val_ratio, seed)  # 按同样规则重新划分数据。
    selected_samples = select_samples(args.split, train_samples, val_samples, test_samples)  # 根据 --split 选择要评估的集合。

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}  # 把类别映射反过来，变成编号到类别名。
    class_names = [idx_to_class[idx] for idx in range(len(idx_to_class))]  # 按编号顺序得到类别名列表，保证矩阵行列顺序正确。
    model = load_model(checkpoint, device)  # 创建模型并加载训练好的权重。
    loader = make_eval_loader(selected_samples, checkpoint, args.batch_size, args.num_workers)  # 创建评估用 DataLoader。
    confusion, accuracy, total_seen = evaluate(model, loader, len(class_names), device)  # 执行评估并得到混淆矩阵。

    args.output_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录。
    csv_path = args.output_dir / f"confusion_matrix_{args.split}.csv"  # 设置混淆矩阵 CSV 保存路径。
    png_path = args.output_dir / f"confusion_matrix_{args.split}.png"  # 设置混淆矩阵图片保存路径。
    save_confusion_csv(csv_path, confusion, class_names)  # 保存混淆矩阵 CSV。
    save_confusion_plot(  # 保存混淆矩阵数量图。
        png_path,  # 图片保存路径。
        confusion,  # 原始数量混淆矩阵。
        class_names,  # 类别名列表。
        title=f"Confusion Matrix ({args.split}, acc={accuracy:.4f})",  # 图片标题，包含评估集合和准确率。
        normalized=False,  # 这里画原始数量，不做归一化。
    )  # 结束数量图保存。

    if args.normalize:  # 如果用户加了 --normalize，就额外保存比例图。
        normalized_png_path = args.output_dir / f"confusion_matrix_{args.split}_normalized.png"  # 设置归一化图片路径。
        save_confusion_plot(  # 保存归一化混淆矩阵图片。
            normalized_png_path,  # 图片保存路径。
            confusion,  # 原始混淆矩阵，函数内部会归一化。
            class_names,  # 类别名列表。
            title=f"Normalized Confusion Matrix ({args.split}, acc={accuracy:.4f})",  # 图片标题。
            normalized=True,  # 开启按行归一化。
        )  # 结束归一化图保存。

    print(f"Device: {device}")  # 打印当前评估设备。
    print(f"Split: {args.split}")  # 打印评估的是 train、val 还是 test。
    print(f"Samples: {total_seen}")  # 打印评估样本数量。
    print(f"Accuracy: {accuracy:.4f}")  # 打印整体准确率。
    print(f"Saved CSV to: {csv_path}")  # 打印 CSV 保存路径。
    print(f"Saved plot to: {png_path}")  # 打印图片保存路径。
    if args.normalize:  # 如果保存了归一化图。
        print(f"Saved normalized plot to: {normalized_png_path}")  # 打印归一化图片保存路径。


if __name__ == "__main__":  # 脚本入口：只有直接运行 evaluate_confusion_matrix.py 时才执行 main。
    main()  # 调用主函数，开始评估并保存混淆矩阵。
