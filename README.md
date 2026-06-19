# AAA_Pobject

一个基于 **PyTorch + torchaudio** 的咀嚼/进食声音分类项目。项目主要功能包括：音频分类模型训练、单个音频预测、混淆矩阵评估，以及训练曲线可视化。

## 项目简介

本项目通过读取不同食物类别的音频文件，将音频转换为 Mel 频谱特征，再使用卷积神经网络 `AudioCNN` 进行分类训练。训练完成后，可以使用保存的模型对新的音频文件进行预测，并生成评估结果和可视化图表。

## 项目文件说明

```text
AAA_Pobject/
├── train_chewing_audio.py          # 训练咀嚼声音分类模型
├── predict_audio.py                # 使用训练好的模型预测单个音频
├── evaluate_confusion_matrix.py    # 生成混淆矩阵并评估模型效果
├── plot_training_curves.py         # 绘制训练 loss 和 accuracy 曲线
├── archive/clips_rd/               # 默认数据集目录，需要用户自行准备
└── runs/chewing_audio/             # 默认训练输出目录
```

## 环境依赖

建议使用 Python 3.9 及以上版本。

需要安装的主要依赖：

```bash
pip install torch torchaudio matplotlib numpy
```

如果使用 NVIDIA 显卡训练，建议根据自己的 CUDA 版本到 PyTorch 官网选择对应安装命令。

## 数据集目录格式

训练脚本默认读取：

```text
archive/clips_rd
```

数据集需要按照“一个类别一个文件夹”的形式存放，例如：

```text
archive/clips_rd/
├── apple/
│   ├── apple_1_001.wav
│   ├── apple_1_002.wav
│   └── ...
├── chips/
│   ├── chips_1_001.wav
│   ├── chips_1_002.wav
│   └── ...
└── bread/
    ├── bread_1_001.wav
    ├── bread_1_002.wav
    └── ...
```

支持的音频格式包括：

```text
.wav, .flac, .mp3, .ogg, .m4a
```

## 模型训练

直接使用默认参数训练：

```bash
python train_chewing_audio.py
```

指定数据集目录、输出目录、训练轮数和 batch size：

```bash
python train_chewing_audio.py --data-dir archive/clips_rd --output-dir runs/chewing_audio --epochs 50 --batch-size 32
```

快速测试脚本能否正常运行：

```bash
python train_chewing_audio.py --limit-per-class 20 --epochs 2
```

如果显卡支持 CUDA，也可以开启混合精度训练：

```bash
python train_chewing_audio.py --amp
```

训练默认参数包括：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--data-dir` | `archive/clips_rd` | 数据集目录 |
| `--output-dir` | `runs/chewing_audio` | 训练结果保存目录 |
| `--epochs` | `50` | 训练轮数 |
| `--batch-size` | `32` | 批次大小 |
| `--lr` | `3e-4` | 学习率 |
| `--sample-rate` | `16000` | 统一音频采样率 |
| `--duration` | `4.0` | 每条音频统一长度，单位秒 |
| `--n-mels` | `96` | Mel 频谱通道数 |
| `--train-ratio` | `0.8` | 训练集比例 |
| `--val-ratio` | `0.1` | 验证集比例 |

训练完成后，默认会在 `runs/chewing_audio/` 下生成：

```text
class_to_idx.json      # 类别名与数字标签的对应关系
split_summary.json     # 训练集、验证集、测试集划分统计
best_model.pt          # 验证集效果最好的模型
last_model.pt          # 最终保存的模型
history.json           # 每一轮训练记录
metrics.json           # 最终评估指标
```

## 单个音频预测

使用默认模型预测一个音频文件：

```bash
python predict_audio.py path/to/audio.wav
```

指定模型路径并显示概率最高的前 5 个类别：

```bash
python predict_audio.py path/to/audio.wav --model runs/chewing_audio/best_model.pt --top-k 5
```

指定运行设备：

```bash
python predict_audio.py path/to/audio.wav --device cpu
python predict_audio.py path/to/audio.wav --device cuda
```

输出示例：

```text
Device: cuda
Audio: path/to/audio.wav
Top predictions:
1. chips: 87.35%
2. apple: 6.42%
3. bread: 3.18%
```

## 模型评估与混淆矩阵

评估测试集并生成混淆矩阵：

```bash
python evaluate_confusion_matrix.py --split test
```

同时生成归一化混淆矩阵：

```bash
python evaluate_confusion_matrix.py --split test --normalize
```

指定数据集、模型和输出目录：

```bash
python evaluate_confusion_matrix.py --data-dir archive/clips_rd --model runs/chewing_audio/best_model.pt --output-dir runs/chewing_audio/evaluation --split test --normalize
```

默认会生成：

```text
runs/chewing_audio/evaluation/confusion_matrix_test.csv
runs/chewing_audio/evaluation/confusion_matrix_test.png
runs/chewing_audio/evaluation/confusion_matrix_test_normalized.png
```

## 绘制训练曲线

根据训练生成的 `history.json` 和 `metrics.json` 绘制 loss、accuracy 曲线：

```bash
python plot_training_curves.py
```

指定训练结果目录：

```bash
python plot_training_curves.py --run-dir runs/chewing_audio
```

默认输出：

```text
runs/chewing_audio/training_curves.png
```

## 核心流程

```text
准备数据集
   ↓
运行 train_chewing_audio.py 训练模型
   ↓
生成 best_model.pt、history.json、metrics.json
   ↓
运行 predict_audio.py 预测新音频
   ↓
运行 evaluate_confusion_matrix.py 评估模型
   ↓
运行 plot_training_curves.py 可视化训练过程
```

## 注意事项

1. 第一次使用前需要先准备好 `archive/clips_rd` 数据集目录。
2. 如果没有 `runs/chewing_audio/best_model.pt`，需要先运行训练脚本。
3. 数据集类别文件夹名称会被作为类别名称，请保持训练、预测、评估时的数据集类别顺序一致。
4. Windows 环境下如果 DataLoader 多进程报错，可以保持 `--num-workers 0`。
5. 如果训练时显存不足，可以减小 `--batch-size`，例如改为 `16` 或 `8`。

## 作者

- GitHub: [1830420924](https://github.com/1830420924)
