"""评估训练好的音频分类模型，并保存混淆矩阵。

这个脚本可以评估 train / val / test 任意一个集合，输出整体准确率，
并把混淆矩阵保存为 CSV 表格和 PNG 图片。
"""

import argparse  # 用于读取命令行参数。
import csv  # 用于保存混淆矩阵 CSV 文件。
from pathlib import Path  # 用 pathlib 管理文件路径。

import matplotlib.pyplot as plt  # 用于绘制混淆矩阵图片。
import numpy as np  # 用于保存和处理混淆矩阵数组。
import torch  # PyTorch：加载模型、GPU 推理、张量计算。
from torch.utils.data import DataLoader  # 批量读取评估数据。

from train_chewing_audio import AudioCNN  # 导入训练时使用的模型结构。
from train_chewing_audio import ChewingSoundDataset  # 导入训练时使用的数据集类，保证预处理一致。
from train_chewing_audio import list_audio_files  # 复用训练脚本的数据扫描逻辑。
from train_chewing_audio import split_samples  # 复用训练脚本的数据划分逻辑。


def parse_args():
    """解析评估脚本的命令行参数。"""
    parser = argparse.ArgumentParser(description="Evaluate a chewing sound classifier and save confusion matrix.")
    parser.add_argument("--data-dir", type=Path, default=Path("archive/clips_rd"))  # 数据集目录。
    parser.add_argument("--model", type=Path, default=Path("runs/chewing_audio/best_model.pt"))  # 要评估的模型文件。
    parser.add_argument("--output-dir", type=Path, default=Path("runs/chewing_audio/evaluation"))  # 评估结果输出目录。
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")  # 选择评估哪个集合。
    parser.add_argument("--batch-size", type=int, default=64)  # 评估 batch size。
    parser.add_argument("--num-workers", type=int, default=0)  # Windows 默认 0 更稳。
    parser.add_argument("--seed", type=int, default=None)  # 不指定时从 checkpoint 读取训练时 seed。
    parser.add_argument("--train-ratio", type=float, default=None)  # 不指定时从 checkpoint 读取训练时 train_ratio。
    parser.add_argument("--val-ratio", type=float, default=None)  # 不指定时从 checkpoint 读取训练时 val_ratio。
    parser.add_argument("--normalize", action="store_true")  # 是否额外保存归一化混淆矩阵。
    return parser.parse_args()


def choose_device():
    """自动选择评估设备：有 CUDA 就用 GPU，否则用 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model_path, device):
    """加载模型 checkpoint。"""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return torch.load(model_path, map_location=device, weights_only=False)


def checkpoint_arg(checkpoint, name, default):
    """从 checkpoint 读取训练参数，兼容新旧 checkpoint 格式。"""
    saved_args = checkpoint.get("args", {})
    return checkpoint.get(name, saved_args.get(name, default))


def resolve_split_config(args, checkpoint):
    """确定数据划分参数。

    评估时必须使用和训练时一致的 seed、train_ratio、val_ratio，
    否则评估集可能和训练时保存的 best_model 不对应。
    """
    seed = args.seed if args.seed is not None else checkpoint_arg(checkpoint, "seed", 42)
    train_ratio = args.train_ratio if args.train_ratio is not None else checkpoint_arg(checkpoint, "train_ratio", 0.8)
    val_ratio = args.val_ratio if args.val_ratio is not None else checkpoint_arg(checkpoint, "val_ratio", 0.1)
    return seed, train_ratio, val_ratio


def check_class_mapping(current_mapping, checkpoint_mapping):
    """检查当前数据集类别编号是否和训练时一致。"""
    if current_mapping != checkpoint_mapping:
        raise RuntimeError(
            "Class mapping mismatch between dataset and checkpoint. "
            "Please evaluate with the same dataset folder used for training."
        )


def select_samples(split_name, train_samples, val_samples, test_samples):
    """根据 split 参数选择要评估的数据集合。"""
    if split_name == "train":
        return train_samples
    if split_name == "val":
        return val_samples
    return test_samples


def load_model(checkpoint, device):
    """根据 checkpoint 重建模型并加载权重。"""
    class_to_idx = checkpoint["class_to_idx"]
    model_size = checkpoint_arg(checkpoint, "model_size", "legacy")  # 旧模型没有 model_size，默认按 legacy 加载。
    dropout = checkpoint_arg(checkpoint, "dropout", 0.35)

    model = AudioCNN(
        num_classes=len(class_to_idx),
        sample_rate=checkpoint["sample_rate"],
        n_mels=checkpoint["n_mels"],
        model_size=model_size,
        dropout=dropout,
        specaugment=False,  # 评估时关闭 SpecAugment，保证结果稳定。
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def make_eval_loader(samples, checkpoint, batch_size, num_workers):
    """创建评估用 DataLoader。"""
    dataset = ChewingSoundDataset(
        samples=samples,
        sample_rate=checkpoint["sample_rate"],
        duration=checkpoint["duration"],
        training=False,  # 评估阶段不做随机增强。
        cache_audio=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def evaluate(model, loader, num_classes, device):
    """遍历数据集，计算整体准确率和混淆矩阵。

    混淆矩阵行表示真实类别，列表示预测类别。
    """
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_correct = 0
    total_seen = 0

    with torch.inference_mode():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(waveforms)
            predictions = logits.argmax(dim=1)

            total_correct += (predictions == labels).sum().item()
            total_seen += labels.numel()

            # 把标签和预测结果转到 CPU，用于更新 numpy 混淆矩阵。
            labels_cpu = labels.cpu().numpy()
            predictions_cpu = predictions.cpu().numpy()
            for true_label, predicted_label in zip(labels_cpu, predictions_cpu):
                confusion[true_label, predicted_label] += 1

    accuracy = total_correct / total_seen if total_seen else 0.0
    return confusion, accuracy, total_seen


def save_confusion_csv(path, confusion, class_names):
    """把混淆矩阵保存为 CSV 文件，方便用 Excel 查看。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + class_names)
        for class_name, row in zip(class_names, confusion):
            writer.writerow([class_name] + row.tolist())


def normalized_confusion(confusion):
    """按行归一化混淆矩阵。

    每一行除以当前真实类别的样本总数，得到每个类别被预测成各类别的比例。
    """
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return confusion / row_sums


def save_confusion_plot(path, confusion, class_names, title, normalized=False):
    """把混淆矩阵绘制并保存成 PNG 图片。"""
    matrix = normalized_confusion(confusion) if normalized else confusion
    fig_size = max(10, len(class_names) * 0.55)  # 类别越多，图片尺寸越大，避免文字挤在一起。
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # 在每个格子里写上数值。深色背景使用白字，浅色背景使用黑字。
    threshold = matrix.max() * 0.6 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            text = f"{value:.2f}" if normalized else str(int(value))
            color = "white" if value > threshold else "black"
            ax.text(col_index, row_index, text, ha="center", va="center", color=color, fontsize=7)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    """评估脚本主流程。"""
    args = parse_args()
    device = choose_device()
    checkpoint = load_checkpoint(args.model, device)
    seed, train_ratio, val_ratio = resolve_split_config(args, checkpoint)

    # 重新扫描并按训练时参数划分数据，确保评估集合和训练时一致。
    samples, class_to_idx = list_audio_files(args.data_dir)
    check_class_mapping(class_to_idx, checkpoint["class_to_idx"])
    train_samples, val_samples, test_samples = split_samples(samples, train_ratio, val_ratio, seed)
    selected_samples = select_samples(args.split, train_samples, val_samples, test_samples)

    # 按类别编号顺序生成类别名列表，保证混淆矩阵行列顺序正确。
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    class_names = [idx_to_class[idx] for idx in range(len(idx_to_class))]

    model = load_model(checkpoint, device)
    loader = make_eval_loader(selected_samples, checkpoint, args.batch_size, args.num_workers)
    confusion, accuracy, total_seen = evaluate(model, loader, len(class_names), device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"confusion_matrix_{args.split}.csv"
    png_path = args.output_dir / f"confusion_matrix_{args.split}.png"

    save_confusion_csv(csv_path, confusion, class_names)
    save_confusion_plot(png_path, confusion, class_names, title=f"Confusion Matrix ({args.split}, acc={accuracy:.4f})")

    # 如果用户加了 --normalize，就额外保存归一化混淆矩阵。
    if args.normalize:
        normalized_png_path = args.output_dir / f"confusion_matrix_{args.split}_normalized.png"
        save_confusion_plot(
            normalized_png_path,
            confusion,
            class_names,
            title=f"Normalized Confusion Matrix ({args.split}, acc={accuracy:.4f})",
            normalized=True,
        )

    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Samples: {total_seen}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved plot to: {png_path}")
    if args.normalize:
        print(f"Saved normalized plot to: {normalized_png_path}")


if __name__ == "__main__":
    main()
