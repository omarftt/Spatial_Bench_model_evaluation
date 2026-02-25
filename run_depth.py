import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def load_model(model_name):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, processor


def run_depth(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth  # (1, H, W)

    # Interpolate back to original resolution
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],  # (H, W)
        mode="bicubic",
        align_corners=False,
    ).squeeze()  # (H, W)

    return prediction.cpu().float().numpy(), image.size  # depth_array, (W, H)


def save_depth(depth_array, output_path, save_colored):
    # Normalize to 0-255 for grayscale PNG
    d_min, d_max = depth_array.min(), depth_array.max()
    normalized = (depth_array - d_min) / (d_max - d_min + 1e-8)
    gray = (normalized * 255).astype(np.uint8)
    Image.fromarray(gray).save(output_path.with_suffix(".png"))

    if save_colored:
        try:
            import matplotlib.cm as cm
            colored = (cm.inferno(normalized)[:, :, :3] * 255).astype(np.uint8)
            Image.fromarray(colored).save(
                output_path.parent / (output_path.stem + "_colored.png")
            )
        except ImportError:
            print("matplotlib not found — skipping colored output", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Run Depth Anything V2 on a folder of images")
    parser.add_argument("--input_dir", required=True, help="Folder containing input images")
    parser.add_argument("--output_dir", required=True, help="Folder to save depth maps")
    parser.add_argument(
        "--model",
        default="depth-anything/Depth-Anything-V2-Large-hf",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--save_colored",
        action="store_true",
        help="Also save a colorized (inferno) depth visualization",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"No images found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_paths)} images. Loading model: {args.model}")
    model, processor = load_model(args.model)
    print("Model loaded. Running depth estimation...")

    for i, img_path in enumerate(image_paths, 1):
        out_path = output_dir / img_path.stem
        try:
            depth_array, _ = run_depth(img_path, model, processor)
            save_depth(depth_array, out_path, args.save_colored)
            print(f"[{i}/{len(image_paths)}] {img_path.name} -> {out_path.with_suffix('.png').name}")
        except Exception as e:
            print(f"[{i}/{len(image_paths)}] ERROR on {img_path.name}: {e}", file=sys.stderr)

    print(f"Done. Depth maps saved to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in run_depth.py: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
