import os
import subprocess
import sys
import tempfile
import json
import argparse
from pathlib import Path
from utils import build_full_prompt, save_results, print_summary, get_inference_script


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
    
    # Auto-detect inference script from scripts/ folder
    INFER_SCRIPT = get_inference_script(MODEL_NAME)
    
    # Verify inference script exists
    if not os.path.exists(INFER_SCRIPT):
        print(f"ERROR: Inference script '{INFER_SCRIPT}' not found")
        sys.exit(1)
    
    if not os.path.exists(JSONL_INPUT):
        print(f"ERROR: JSONL file '{JSONL_INPUT}' not found")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    predictions_file = os.path.join(OUTPUT_DIR, "predictions.jsonl")
    summary_file = os.path.join(OUTPUT_DIR, "summary.json")

    with open(JSONL_INPUT, 'r', encoding='utf-8') as f:
        samples = [json.loads(line) for line in f if line.strip()]
    
    print(f"Model: {MODEL_NAME}")
    print(f"Inference script: {INFER_SCRIPT}")
    print(f"Processing {len(samples)} samples...\n")
    
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
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp_file:
            tmp_file.write(full_prompt)
            prompt_file = tmp_file.name
        
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp_out:
            output_file = tmp_out.name
        
        try:
            cmd = [
                sys.executable, INFER_SCRIPT,
                "--images"] + image_paths + [
                "--prompt_file", prompt_file,
                "--output_file", output_file,
                "--model", MODEL_NAME,
                "--max_new_tokens", str(MAX_NEW_TOKENS),
            ]
            
            subprocess.run(cmd, check=True, text=True, capture_output=True, encoding="utf-8")
            
            with open(output_file, 'r', encoding='utf-8') as f:
                prediction = f.read().strip()
                
        except subprocess.CalledProcessError as e:
            prediction = f"ERROR: {e.stderr if e.stderr else str(e)}"
        except Exception as e:
            prediction = f"ERROR: {e}"
        finally:
            if os.path.exists(prompt_file):
                os.remove(prompt_file)
            if os.path.exists(output_file):
                os.remove(output_file)
        
        sample['prediction'] = prediction
        results.append(sample)
        
        print(f"[{idx}/{len(samples)}] {sample.get('skill', 'unknown')}: {prediction}")
        
        save_results(results, predictions_file, summary_file)
    
    summary = save_results(results, predictions_file, summary_file)
    print_summary(summary)
    print(f"\nSaved: {predictions_file}, {summary_file}")


if __name__ == "__main__":
    main()