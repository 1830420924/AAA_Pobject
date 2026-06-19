import argparse  # 导入 argparse：用来读取命令行参数，比如 history.json 在哪里、图片保存到哪里。
import json  # 导入 json：用来读取训练脚本保存的 history.json 和 metrics.json。
from pathlib import Path  # 导入 Path：用更方便、更清晰的方式处理文件路径。

import matplotlib.pyplot as plt  # 导入 matplotlib 的绘图模块，用来画 loss 和 accuracy 曲线。


def parse_args():  # 定义命令行参数解析函数。
    parser = argparse.ArgumentParser(description="Plot training loss and accuracy curves.")  # 创建参数解析器，并写一句脚本说明。
    parser.add_argument("--run-dir", type=Path, default=Path("runs/chewing_audio"))  # 训练输出目录，默认读取 runs/chewing_audio。
    parser.add_argument("--history", type=Path, default=None)  # 训练历史文件路径；不填时默认使用 run-dir/history.json。
    parser.add_argument("--metrics", type=Path, default=None)  # 最终指标文件路径；不填时默认使用 run-dir/metrics.json。
    parser.add_argument("--output", type=Path, default=None)  # 图片输出路径；不填时默认保存为 run-dir/training_curves.png。
    parser.add_argument("--dpi", type=int, default=200)  # 图片清晰度，数值越高图片越清晰，文件也会稍大。
    return parser.parse_args()  # 解析命令行参数，并返回给主程序使用。


def load_json(path):  # 定义读取 JSON 文件的小工具函数。
    if not path.exists():  # 如果文件不存在，就不能继续读取。
        raise FileNotFoundError(f"File not found: {path}")  # 抛出清楚的错误，告诉用户哪个文件没找到。
    with path.open("r", encoding="utf-8") as handle:  # 使用 UTF-8 编码打开 JSON 文件。
        return json.load(handle)  # 把 JSON 内容读取成 Python 对象并返回。


def pick_paths(args):  # 根据用户参数决定实际读取和保存的文件路径。
    history_path = args.history if args.history is not None else args.run_dir / "history.json"  # 如果用户没指定 history，就用默认路径。
    metrics_path = args.metrics if args.metrics is not None else args.run_dir / "metrics.json"  # 如果用户没指定 metrics，就用默认路径。
    output_path = args.output if args.output is not None else args.run_dir / "training_curves.png"  # 如果用户没指定 output，就保存到默认图片路径。
    return history_path, metrics_path, output_path  # 返回三个实际路径。


def extract_series(history):  # 从 history.json 中提取画图需要的几个列表。
    epochs = [row["epoch"] for row in history]  # 提取每一轮的 epoch 编号，作为横轴。
    train_loss = [row["train_loss"] for row in history]  # 提取训练集 loss 曲线。
    val_loss = [row["val_loss"] for row in history]  # 提取验证集 loss 曲线。
    train_acc = [row["train_acc"] * 100 for row in history]  # 提取训练集准确率，并从 0-1 转成百分比。
    val_acc = [row["val_acc"] * 100 for row in history]  # 提取验证集准确率，并从 0-1 转成百分比。
    return epochs, train_loss, val_loss, train_acc, val_acc  # 返回所有曲线数据。


def best_validation_point(history):  # 找出验证集准确率最高的那一轮，用来在图上标记星号。
    best_row = max(history, key=lambda row: row["val_acc"])  # 按 val_acc 找最大值对应的记录。
    best_epoch = best_row["epoch"]  # 取出最佳验证准确率对应的轮数。
    best_val_acc = best_row["val_acc"] * 100  # 把最佳验证准确率转成百分比。
    return best_epoch, best_val_acc  # 返回最佳轮数和最佳准确率。


def plot_curves(history, metrics, output_path, dpi):  # 根据训练历史和指标绘制曲线图。
    epochs, train_loss, val_loss, train_acc, val_acc = extract_series(history)  # 提取横轴和四条曲线的数据。
    best_epoch, best_val_acc = best_validation_point(history)  # 找到验证准确率最高的位置。
    test_acc = metrics.get("test_acc")  # 从 metrics.json 中读取测试集准确率，可能不存在所以用 get。

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)  # 创建上下两个子图，共用同一个 epoch 横轴。
    fig.suptitle("Chewing Sound Classification Training Curves", fontsize=14)  # 给整张图加一个总标题。

    axes[0].plot(epochs, train_loss, label="Train Loss", color="#2a6f97", linewidth=1.8)  # 在上方子图画训练 loss。
    axes[0].plot(epochs, val_loss, label="Val Loss", color="#b56576", linewidth=1.8, linestyle="--")  # 在上方子图画验证 loss。
    axes[0].set_title("Training and Validation Loss Records")  # 设置上方子图标题。
    axes[0].set_ylabel("Loss")  # 设置上方子图纵轴标签。
    axes[0].grid(True, alpha=0.28)  # 添加淡淡的网格线，让曲线更容易读。
    axes[0].legend()  # 显示图例，说明哪条线代表 train loss 或 val loss。

    axes[1].plot(epochs, train_acc, label="Train Accuracy", color="#2a6f97", linewidth=1.8)  # 在下方子图画训练准确率。
    axes[1].plot(epochs, val_acc, label="Val Accuracy", color="#b56576", linewidth=1.8, linestyle="--")  # 在下方子图画验证准确率。
    axes[1].scatter([best_epoch], [best_val_acc], marker="*", s=160, color="#6d597a", label=f"Best Val: {best_val_acc:.2f}%")  # 用星号标出验证集最好的一轮。
    if test_acc is not None:  # 如果 metrics.json 中有测试集准确率，就在图中额外显示。
        axes[1].axhline(test_acc * 100, color="#4d908e", linestyle=":", linewidth=1.4, label=f"Test Acc: {test_acc * 100:.2f}%")  # 用横向虚线标出测试准确率。
    axes[1].set_title("Training and Validation Accuracy Records")  # 设置下方子图标题。
    axes[1].set_xlabel("Epoch")  # 设置横轴标签。
    axes[1].set_ylabel("Accuracy (%)")  # 设置纵轴标签，单位是百分比。
    axes[1].set_ylim(0, 100)  # 准确率范围固定为 0 到 100，更符合直觉。
    axes[1].grid(True, alpha=0.28)  # 添加淡淡的网格线。
    axes[1].legend()  # 显示准确率曲线图例。

    fig.tight_layout(rect=(0, 0, 1, 0.96))  # 自动调整布局，同时给总标题留一点空间。
    output_path.parent.mkdir(parents=True, exist_ok=True)  # 确保输出目录存在，不存在就自动创建。
    fig.savefig(output_path, dpi=dpi)  # 把图保存成 PNG 文件。
    plt.close(fig)  # 关闭图像对象，释放内存。


def main():  # 主函数，组织完整绘图流程。
    args = parse_args()  # 读取命令行参数。
    history_path, metrics_path, output_path = pick_paths(args)  # 确定 history、metrics 和输出图片的路径。
    history = load_json(history_path)  # 读取每一轮训练记录。
    metrics = load_json(metrics_path) if metrics_path.exists() else {}  # 如果 metrics.json 存在就读取，否则使用空字典。
    plot_curves(history, metrics, output_path, args.dpi)  # 调用绘图函数生成训练曲线图片。
    print(f"Loaded history from: {history_path}")  # 打印 history.json 路径，方便确认读取的是哪个实验。
    print(f"Loaded metrics from: {metrics_path if metrics_path.exists() else 'not found'}")  # 打印 metrics.json 路径，如果没有就显示 not found。
    print(f"Saved training curves to: {output_path}")  # 打印图片保存位置。


if __name__ == "__main__":  # Python 脚本入口，只有直接运行本文件时才执行 main。
    main()  # 调用主函数，开始绘图。
