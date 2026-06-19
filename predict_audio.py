import argparse
from pathlib import Path

import torch
import torchaudio

from train_chewing_audio import AudioCNN


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one chewing sound audio file.")
    parser.add_argument("audio", type=Path, help="Path to one audio file, such as a .wav file.")
    parser.add_argument("--model", type=Path, default=Path("runs/chewing_audio/best_model.pt"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def choose_device(device_arg):
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("You requested CUDA, but CUDA is not available.")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(model_path, device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return torch.load(model_path, map_location=device, weights_only=False)


def checkpoint_arg(checkpoint, name, default):
    saved_args = checkpoint.get("args", {})
    return checkpoint.get(name, saved_args.get(name, default))


def load_model(checkpoint, device):
    class_to_idx = checkpoint["class_to_idx"]
    sample_rate = checkpoint["sample_rate"]
    n_mels = checkpoint["n_mels"]
    model_size = checkpoint_arg(checkpoint, "model_size", "legacy")
    dropout = checkpoint_arg(checkpoint, "dropout", 0.35)

    model = AudioCNN(
        num_classes=len(class_to_idx),
        sample_rate=sample_rate,
        n_mels=n_mels,
        model_size=model_size,
        dropout=dropout,
        specaugment=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def crop_or_pad(waveform, target_samples):
    length = waveform.shape[-1]
    if length > target_samples:
        start = (length - target_samples) // 2
        waveform = waveform[:, start : start + target_samples]
    elif length < target_samples:
        pad = target_samples - length
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    return waveform


def load_audio(audio_path, sample_rate, duration, device):
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    waveform, source_rate = torchaudio.load(str(audio_path))
    waveform = waveform.mean(dim=0, keepdim=True)
    if source_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
    peak = waveform.abs().max().clamp_min(1e-6)
    waveform = waveform / peak
    target_samples = int(sample_rate * duration)
    waveform = crop_or_pad(waveform, target_samples)
    waveform = waveform.unsqueeze(0)
    return waveform.to(device)


def predict(model, waveform, top_k):
    with torch.inference_mode():
        logits = model(waveform)
        probabilities = torch.softmax(logits, dim=1)
        top_probs, top_indices = probabilities.topk(top_k, dim=1)
    return top_probs[0].cpu(), top_indices[0].cpu()


def main():
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = load_checkpoint(args.model, device)
    model = load_model(checkpoint, device)
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
