import argparse
import sys
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from prompts import SYSTEM_PROMPT


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):

    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Calculate possible tiling configurations
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize image to target size
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    
    # Split into tiles
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    assert len(processed_images) == blocks
    
    # Add thumbnail for global context
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images


def load_and_preprocess(image_path, input_size=448, max_num=12):
    """Load and dynamically preprocess image."""
    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def run_inference(image_paths, prompt_file, output_file, model_name, max_new_tokens):
    """Run inference on multiple images with the given prompt."""
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load images with dynamic preprocessing
    pixel_values_list = []
    num_patches_list = []
    
    for img_path in image_paths:
        pixel_values = load_and_preprocess(img_path, max_num=12)
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.size(0))
    
    # Concatenate all tiles
    pixel_values = torch.cat(pixel_values_list, dim=0)
    
    if device == "cuda":
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
    
    # Build question with proper image tokens and system prompt
    num_images = len(image_paths)
    image_tokens = ''.join([f'Image-{i+1}: <image>\n' for i in range(num_images)])
    
    # Add system prompt at the beginning of the question
    question = f"{SYSTEM_PROMPT}\n\n{image_tokens}\n{prompt}"
    
    generation_config = {
        'max_new_tokens': max_new_tokens,
        'do_sample': False,
    }
    
    # Pass num_patches_list to preserve image boundaries
    response = model.chat(tokenizer, pixel_values, question, generation_config, 
                         num_patches_list=num_patches_list)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(response.strip())


def main():
    parser = argparse.ArgumentParser(description="Multi-agent VQA with InternVL2 models")
    parser.add_argument("--images", nargs='+', required=True)
    parser.add_argument("--prompt_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--model", default="OpenGVLab/InternVL2-8B")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()
    
    run_inference(
        image_paths=args.images,
        prompt_file=args.prompt_file,
        output_file=args.output_file,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR in infer_internvl.py: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)