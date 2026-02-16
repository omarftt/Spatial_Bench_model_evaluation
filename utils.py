import json
from pathlib import Path


def get_inference_script(model_name):
    """Map model name to appropriate inference script in scripts/ folder."""
    model_lower = model_name.lower()
    
    # Qwen models
    if "qwen2.5-vl" in model_lower or "qwen2-vl" in model_lower:
        return "scripts/infer_qwen.py"
    
    # LLaVA and VILA models
    elif "llava" in model_lower or "vila" in model_lower:
        return "scripts/infer_llava.py"
    
    # InternVL models
    elif "internvl" in model_lower:
        return "scripts/infer_internvl.py"
    
    # Molmo models
    elif "molmo" in model_lower:
        return "scripts/infer_molmo.py"

    
    # Default fallback
    else:
        raise ValueError(
            f"Unknown model family: {model_name}\n"
        )


def format_choices(choices):
    """Format choices"""
    labels = ['A', 'B', 'C', 'D']
    formatted = []
    for i, choice in enumerate(choices):
        if i < len(labels):
            formatted.append(f"{labels[i]}) {choice}")
    return "\n".join(formatted)


def build_full_prompt(question, choices):
    """Build the complete prompt"""
    formatted_choices = format_choices(choices)
    full_prompt = f"{question}\n\nChoices:\n{formatted_choices}"
    return full_prompt


def extract_answer_letter(prediction):
    """Extract the letter from prediction"""
    pred_upper = prediction.strip().upper()
    for letter in ['A', 'B', 'C', 'D']:
        if pred_upper.startswith(f"{letter})") or pred_upper.startswith(letter):
            return letter
    return None


def ground_truth_to_letter(ground_truth, choices):
    """Convert ground truth text to letter based on choices"""
    try:
        index = choices.index(ground_truth)
        return chr(65 + index) 
    except (ValueError, IndexError):
        return None


def compute_summary(results):
    """Compute accuracy summary by skill and overall"""
    total_correct = 0
    total_samples = 0
    by_skill = {}
    
    for result in results:
        if result.get('prediction', '').startswith('ERROR'):
            continue
            
        skill = result.get('skill', 'unknown')
        gt_text = result.get('ground_truth', '')
        choices = result.get('choices', [])
        prediction = result.get('prediction', '')
        
        gt_letter = ground_truth_to_letter(gt_text, choices)
        pred_letter = extract_answer_letter(prediction)
        
        if gt_letter is None or pred_letter is None:
            continue
        
        is_correct = (pred_letter == gt_letter)
        
        if skill not in by_skill:
            by_skill[skill] = {'correct': 0, 'total': 0}
        
        by_skill[skill]['total'] += 1
        by_skill[skill]['correct'] += int(is_correct)
        
        total_samples += 1
        total_correct += int(is_correct)
    
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    per_skill_acc = {
        skill: stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        for skill, stats in by_skill.items()
    }
    
    macro_acc = sum(per_skill_acc.values()) / len(per_skill_acc) if per_skill_acc else 0.0
    
    summary = {
        "overall": {
            "accuracy": overall_acc,
            "correct": total_correct,
            "total": total_samples
        },
        "macro_skill_accuracy": macro_acc,
        "per_skill": {
            skill: {
                "accuracy": per_skill_acc[skill],
                "correct": by_skill[skill]['correct'],
                "total": by_skill[skill]['total']
            }
            for skill in sorted(by_skill.keys())
        }
    }
    
    return summary


def save_results(results, predictions_file, summary_file):
    """Save predictions JSONL and summary JSON"""
    # Ensure output directory exists
    Path(predictions_file).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    summary = compute_summary(results)
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, indent=2, fp=f)
    
    return summary


def print_summary(summary):
    """Print summary statistics"""
    print(f"\n{'='*60}")
    print(f"Overall: {summary['overall']['accuracy']:.2%} ({summary['overall']['correct']}/{summary['overall']['total']})")
    print(f"Macro: {summary['macro_skill_accuracy']:.2%}")
    for skill, stats in summary['per_skill'].items():
        print(f"  {skill}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    print(f"{'='*60}")