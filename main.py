import os
import sys
import json
import argparse
import gc
from pathlib import Path
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from utils import build_full_prompt, save_results, print_summary, check_multi_gpu, clear_cuda_cache, log_cuda_memory_stats


def run_inference(image_paths, prompt, model, processor, inference_fn, max_tokens, model_name):
    model_lower = model_name.lower()

    # InternVL and Gemini expect image paths, not PIL images
    if "internvl" in model_lower or "gemini" in model_lower:
        images = image_paths
    else:
        # Other models expect PIL images
        images = [Image.open(img).convert("RGB") for img in image_paths]

    # Run model-specific inference function
    prediction = inference_fn(model, processor, images, prompt, max_tokens)

    # Memory cleanup
    if "internvl" not in model_lower and "gemini" not in model_lower:
        del images
    gc.collect()

    return prediction


def cleanup_model(model):
    del model
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", type=str, default="data/bench_full.jsonl",
                       help="Input jsonl file")
    parser.add_argument("--images", type=str, default="data/images",
                       help="Directory containing images")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct",
                       help="Model name from HuggingFace")
    parser.add_argument("--output", type=str, default="outputs",
                       help="Output directory for results")
    parser.add_argument("--max_tokens", type=int, default=128,
                       help="Maximum new tokens for generation")
    
    args = parser.parse_args()
    
    JSONL_INPUT = args.input
    IMAGE_FOLDER = args.images
    MODEL_NAME = args.model
    OUTPUT_DIR = args.output
    MAX_NEW_TOKENS = args.max_tokens

    if not os.path.exists(JSONL_INPUT):
        print(f"ERROR: JSONL file '{JSONL_INPUT}' not found")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    predictions_file = os.path.join(OUTPUT_DIR, "predictions.jsonl")
    summary_file = os.path.join(OUTPUT_DIR, "summary.json")

    with open(JSONL_INPUT, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"Model: {MODEL_NAME}")
    print(f"Processing {len(samples)} samples...\n")

    # Load model once
    print("Loading model...")
    model_lower = MODEL_NAME.lower()
    multi_gpu = check_multi_gpu()
    if multi_gpu and TORCH_AVAILABLE:
        print(f"Multi-GPU detected: {torch.cuda.device_count()} GPUs")

    # Import and load based on model type
    if "qwen2.5-vl" in model_lower or "qwen2-vl" in model_lower:
        from scripts.infer_qwen import load_qwen_model, prepare_qwen_inputs, generate_qwen
        model, processor = load_qwen_model(MODEL_NAME, multi_gpu)
        inference_fn = lambda m, p, imgs, prompt, max_tok: generate_qwen(m, prepare_qwen_inputs(imgs, prompt, p, m), max_tok)

    elif "llava" in model_lower or "vila" in model_lower:
        from scripts.infer_llava import load_llava_model, prepare_llava_inputs, generate_llava
        model, processor = load_llava_model(MODEL_NAME, multi_gpu)
        inference_fn = lambda m, p, imgs, prompt, max_tok: generate_llava(m, prepare_llava_inputs(imgs, prompt, p, m), max_tok)

    elif "internvl" in model_lower:
        from scripts.infer_internvl import load_internvl_model, prepare_internvl_inputs, generate_internvl
        model, processor = load_internvl_model(MODEL_NAME, multi_gpu)
        inference_fn = lambda m, p, imgs, prompt, max_tok: generate_internvl(m, p, *prepare_internvl_inputs(imgs), prompt, max_tok)

    elif "molmo" in model_lower:
        from scripts.infer_molmo import load_molmo_model, prepare_molmo_inputs, generate_molmo
        model, processor = load_molmo_model(MODEL_NAME, multi_gpu)
        inference_fn = lambda m, p, imgs, prompt, max_tok: generate_molmo(m, prepare_molmo_inputs(imgs, prompt, p, m), max_tok, p)

    elif "gemini" in model_lower:
        # Gemini is just an API call, no model loading needed
        model, processor = None, None
        def gemini_inference(m, p, imgs, prompt, max_tok):
            from scripts.infer_gemini import run_inference as gemini_run_inference
            import tempfile
            # Write prompt to temp file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write(prompt)
                prompt_file = f.name
            # Create temp output file
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                output_file = f.name
            try:
                gemini_run_inference(imgs, prompt_file, output_file, MODEL_NAME, max_tok)
                with open(output_file, 'r') as f:
                    result = f.read().strip()
                return result
            finally:
                os.remove(prompt_file)
                os.remove(output_file)
        inference_fn = gemini_inference

    else:
        print(f"ERROR: Unknown model family: {MODEL_NAME}")
        sys.exit(1)

    if TORCH_AVAILABLE and model and hasattr(model, 'hf_device_map'):
        print(f"Device map: {model.hf_device_map}")
    if TORCH_AVAILABLE and model:
        log_cuda_memory_stats("After loading")
    results = []
    for idx, sample in enumerate(samples, 1):
        images = sample.get('images', [])
        question = sample.get('question', '')
        choices = sample.get('choices', [])
        
        full_prompt = build_full_prompt(question, choices)
        image_paths = [os.path.join(IMAGE_FOLDER, img) for img in images]
        
        missing_images = [img for img in image_paths if not os.path.exists(img)]
        if missing_images:
            sample['prediction'] = "ERROR: Missing images"
            results.append(sample)
            save_results(results, predictions_file, summary_file)
            continue

        try:
            prediction = run_inference(
                image_paths, full_prompt, model, processor,
                inference_fn, MAX_NEW_TOKENS, MODEL_NAME
            )

            # Memory cleanup
            if idx % 5 == 0:
                clear_cuda_cache()

            # Periodic memory logging
            if TORCH_AVAILABLE and idx % 25 == 0:
                log_cuda_memory_stats(f"After sample {idx}")

        except Exception as e:
            if TORCH_AVAILABLE and "CUDA out of memory" in str(e):
                prediction = f"ERROR: CUDA OOM - {e}"
                log_cuda_memory_stats("OOM Event")
                clear_cuda_cache()
            else:
                prediction = f"ERROR: {type(e).__name__} - {str(e)}"
        
        sample['prediction'] = prediction
        results.append(sample)
        
        print(f"[{idx}/{len(samples)}] {sample.get('skill', 'unknown')}: {prediction}")

        save_results(results, predictions_file, summary_file)

    # Cleanup
    if model is not None:
        print("\nCleaning up...")
        cleanup_model(model)
        if TORCH_AVAILABLE:
            log_cuda_memory_stats("Final")

    summary = save_results(results, predictions_file, summary_file)
    print_summary(summary)
    print(f"\nSaved: {predictions_file}, {summary_file}")


if __name__ == "__main__":
    main()