from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from dendritic.enhancement import enhance_model_with_dendritic, get_polynomial_stats
from dendritic.layer import DendriticLayer, DendriticStack
from tqdm import tqdm
import time
import datetime
import json
import os
import psutil
import torch.distributed as dist

# =====================
# Experiment Tracking
# =====================
def get_gpu_mem_usage():
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.max_memory_allocated() / 1024**3  # Return in GB

class ExperimentTracker:
    def __init__(self, method_name, params):
        self.start_time = time.time()
        self.method_name = method_name
        self.params = params
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        self.results = {
            "experiment_id": f"{method_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.datetime.now().isoformat(),
            "method": method_name,
            "params": params,
            "resources": {},
            "metrics": {},
            "training_samples": []
        }
    
    def add_metric(self, name, value, step=None):
        """Add a metric, optionally at a specific step."""
        if step is not None:
            if "training_samples" not in self.results:
                self.results["training_samples"] = []
            self.results["training_samples"].append({
                "step": step,
                "time_sec": time.time() - self.start_time,
                name: value
            })
        else:
            self.results["metrics"][name] = value
    
    def finalize(self, model):
        """Save experiment results to file."""
        # Record final stats
        self.results["resources"]["total_time_min"] = (time.time() - self.start_time) / 60
        self.results["resources"]["peak_gpu_mem_gb"] = get_gpu_mem_usage()
        
        # Parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.results["resources"]["total_params"] = total_params
        self.results["resources"]["trainable_params"] = trainable_params
        
        # Save to file
        filename = os.path.join(self.results_dir, f"{self.results['experiment_id']}.json")
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        return self.results


# =====================
# Experiment Constants
# =====================
TRAINING_STEPS = 6000
GRADIENT_ACCUMULATION_STEPS = 1
BATCH_SIZE = 16
EVAL_INTERVAL = 300
MAX_LENGTH = 256
POLY_RANK = 32

# Sampling function (moved to top for visibility)
def sample_model_output(model, tokenizer, prompt, device, max_new_tokens=64):
    print("\n--- SAMPLE OUTPUT ---")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            temperature=0.8
        )
    generated = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    # print(f"Prompt:\n{prompt}\n")
    print(f"Model output>>>\n{generated}")
    print("--- END SAMPLE ---\n")

print("="*70)
print("DENDRITIC FINETUNING EXPERIMENT (FIXED)")
print("="*70)

# 1. Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# 2. Load dataset
print("\nLoading dataset...")
dataset = load_dataset('iamtarun/python_code_instructions_18k_alpaca', split='train')
dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"Train size: {len(dataset['train'])}")
print(f"Test size: {len(dataset['test'])}")

# 3. Tokenize with PROPER MASKING
def tokenize_function(examples):
    """
    Tokenize with proper loss masking: only compute loss on the output tokens,
    not the prompt tokens.
    """
    # {'instruction': 'Design a python code to convert a given sentence to camelCase', 'input': '', 'output': 'def toCamelCase(s):\n    s = s.split(\' \')\n    return \'\'.join(x.title() for x in s)\n\ns = "this is some random text"\nprint(toCamelCase(s))', 'prompt': 'Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nDesign a python code to convert a given sentence to camelCase\n\n### Input:\n\n\n### Output:\ndef toCamelCase(s):\n    s = s.split(\' \')\n    return \'\'.join(x.title() for x in s)\n\ns = "this is some random text"\nprint(toCamelCase(s))'}
    prompts = examples['prompt']
    outputs = examples['output']

    # For each prompt, extract up to and including '### Output:'
    truncated_prompts = []
    for prompt in prompts:
        idx = prompt.find('### Output:')
        if idx == -1:
            # Fallback: use full prompt
            truncated_prompts.append(prompt)
        else:
            # Include '### Output:' marker
            end_idx = idx + len('### Output:')
            truncated_prompts.append(prompt[:end_idx])

    # Tokenize prompts and outputs separately
    prompt_tokens = tokenizer(truncated_prompts, truncation=False, padding=False)
    output_tokens = tokenizer(
        [output + tokenizer.eos_token for output in outputs],
        truncation=False,
        padding=False
    )

    input_ids = []
    labels = []

    for prompt_ids, output_ids in zip(prompt_tokens['input_ids'], output_tokens['input_ids']):
        # Concatenate prompt + output
        full_ids = prompt_ids + output_ids

        # Truncate if needed
        if len(full_ids) > 256:
            full_ids = full_ids[:256]

        # Create labels: -100 for prompt tokens, actual ids for output tokens
        prompt_len = len(prompt_ids)
        full_labels = [-100] * prompt_len + output_ids

        # Truncate labels too
        if len(full_labels) > 256:
            full_labels = full_labels[:256]

        # Pad to max_length
        padding_length = 256 - len(full_ids)
        full_ids = full_ids + [tokenizer.pad_token_id] * padding_length
        full_labels = full_labels + [-100] * padding_length

        input_ids.append(full_ids)
        labels.append(full_labels)

    return {
        'input_ids': torch.tensor(input_ids),
        'attention_mask': torch.tensor([[1 if id != tokenizer.pad_token_id else 0 for id in ids] for ids in input_ids]),
        'labels': torch.tensor(labels)
    }

print("\nTokenizing with proper masking...")
train_dataset = dataset['train'].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset['train'].column_names,
    batch_size=100  # Process in batches for efficiency
)
eval_dataset = dataset['test'].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset['test'].column_names,
    batch_size=100
)

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Verify masking
print("\nVerifying label masking...")
sample = train_dataset[0]
prompt_tokens = (sample['labels'] == -100).sum().item()
output_tokens = (sample['labels'] != -100).sum().item()
print(f"  Sample 0: {prompt_tokens} prompt tokens masked, {output_tokens} output tokens")
if prompt_tokens == 0:
    print("  ⚠️ WARNING: No tokens masked! Loss will be computed on prompts too.")

# 4. Load models
print("\nLoading models...")
model_base = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
model_dendritic = GPT2LMHeadModel.from_pretrained('gpt2').to(device)

# 5. Enhance with freeze_linear=True from the start
print("\nEnhancing with dendritic layers...")
model_dendritic = enhance_model_with_dendritic(
    model_dendritic,
    target_layers=['mlp.c_fc'],
    poly_rank=POLY_RANK,
    freeze_linear=True,
    verbose=True,
    dendritic_cls=DendriticStack,
    dendritic_kwargs={"dropout": 0.1}
)

# Verify parameter counts
trainable = sum(p.numel() for p in model_dendritic.parameters() if p.requires_grad)
total = sum(p.numel() for p in model_dendritic.parameters())
print(f"\nParameter verification:")
print(f"  Total parameters:     {total:,}")
print(f"  Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")

# expected_total = 124_439_808 + 2_654_220
# expected_trainable = 2_654_220

# if abs(total - expected_total) < 100_000:
#     print(f"  ✓ Total matches expected (~{expected_total:,})")
# else:
#     print(f"  ⚠️ Total mismatch: expected ~{expected_total:,}, got {total:,}")

# if abs(trainable - expected_trainable) < 100_000:
#     print(f"  ✓ Trainable matches expected (~{expected_trainable:,})")
# else:
#     print(f"  ⚠️ Trainable mismatch: expected ~{expected_trainable:,}, got {trainable:,}")

# 7. Evaluation function
def evaluate(model, dataloader, max_batches=None):
    """Evaluate perplexity on a dataset (only on non-masked tokens)."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    nan_batches = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break

            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                labels=labels
            )

            # Count only non-masked tokens
            non_masked = (labels != -100).sum().item()
            batch_loss = outputs.loss.item()

            if not torch.isfinite(outputs.loss):
                print(f"[evaluate] Batch {i}: loss is not finite! loss={batch_loss}, non_masked={non_masked}")
                print(f"  labels: {labels}")
                nan_batches += 1
                continue
            if non_masked == 0:
                print(f"[evaluate] Batch {i}: all tokens masked! Skipping batch.")
                nan_batches += 1
                continue

            total_loss += batch_loss * non_masked
            total_tokens += non_masked

    if total_tokens == 0:
        print("[evaluate] ERROR: No non-masked tokens found in evaluation! Returning nan.")
        return float('nan')

    mean_loss = total_loss / total_tokens
    if not torch.isfinite(torch.tensor(mean_loss)):
        print(f"[evaluate] ERROR: mean_loss is not finite: {mean_loss}")
        return float('nan')
    perplexity = torch.exp(torch.tensor(mean_loss))

    if not torch.isfinite(perplexity):
        print(f"[evaluate] ERROR: perplexity is not finite: {perplexity}")
        return float('nan')

    if nan_batches > 0:
        print(f"[evaluate] WARNING: {nan_batches} batches were skipped due to nan/inf loss or all tokens masked.")

    return perplexity

# Quick sanity check
print("\nQuick eval (100 batches):")
quick_ppl = evaluate(model_dendritic, eval_dataloader, max_batches=100)
print(f"Dendritic (pre-training): {quick_ppl:.2f}")

# 8. Baseline evaluation
print("\n" + "="*70)
print("BASELINE EVALUATION")
print("="*70)
baseline_ppl = evaluate(model_base, eval_dataloader)
print(f"Baseline perplexity: {baseline_ppl:.2f}")

# Verify identity initialization
diff = abs(baseline_ppl - quick_ppl)
if diff < 1.0:
    print(f"✓ Identity initialization verified (diff={diff:.2f})")
else:
    print(f"⚠️ WARNING: Large initialization difference (diff={diff:.2f})")

# 9. Training
print("\n" + "="*70)
print("TRAINING (3000 steps with validation)")
print("="*70)

optimizer = torch.optim.AdamW(
    [p for p in model_dendritic.parameters() if p.requires_grad],
    lr=5e-5,
    weight_decay=0.01
)

print(f"Optimizing {trainable:,} parameters")

model_dendritic.train()
best_eval_ppl = float('inf')
accumulated_loss = 0.0

train_iter = iter(train_dataloader)
tqdm_bar = tqdm(range(TRAINING_STEPS), desc="Training", ncols=100)
for step in tqdm_bar:
    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_dataloader)
        batch = next(train_iter)

    inputs = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    outputs = model_dendritic(input_ids=inputs, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS  # Normalize loss
    loss.backward()
    accumulated_loss += loss.item()

    # Only update weights every GRADIENT_ACCUMULATION_STEPS
    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
        torch.nn.utils.clip_grad_norm_([p for p in model_dendritic.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad()
        tqdm_bar.set_postfix({"avg_loss": f"{accumulated_loss:.4f}"})
        accumulated_loss = 0.0

    # Print stats every 100 steps (after accumulation)
    if (step + 1) % 100 == 0:
        stats = get_polynomial_stats(model_dendritic)
        scales = [s['scale'] for s in stats.values()]
        avg_scale = sum(scales) / len(scales)
        min_scale = min(scales)
        max_scale = max(scales)
        tqdm.write(f"Step {step+1:4d}: scale: avg={avg_scale:+.6f}, min={min_scale:+.6f}, max={max_scale:+.6f}, step loss={loss.item():.4f}")

    # Periodic evaluation (after accumulation)
    if (step + 1) % EVAL_INTERVAL == 0 and (step + 1) > 0:
        eval_ppl = evaluate(model_dendritic, eval_dataloader, max_batches=200)
        tqdm.write(f"  -> Eval perplexity: {eval_ppl:.2f}")
        if eval_ppl < best_eval_ppl:
            best_eval_ppl = eval_ppl
        # Sample output at eval interval
        sample_prompt = "### Instruction:\nWrite a python function to calculate factorial.\n\n### Input:\n\n\n### Output:\n"
        sample_model_output(model_dendritic, tokenizer, sample_prompt, device)
        model_dendritic.train()

# 10. Final evaluation
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)

final_eval_start = time.time()
final_ppl = evaluate(model_dendritic, eval_dataloader)
final_eval_time = time.time() - final_eval_start

print(f"Final evaluation took {final_eval_time:.1f} seconds")
print(f"\nBaseline perplexity:     {baseline_ppl:.2f}")
print(f"Dendritic perplexity:    {final_ppl:.2f}")
print(f"Best eval during train:  {best_eval_ppl:.2f}")

improvement = baseline_ppl - final_ppl
improvement_pct = 100 * improvement / baseline_ppl if baseline_ppl != 0 else 0.0

if improvement > 0:
    print(f"Improvement:             {improvement:.2f} ({improvement_pct:.1f}%)")
else:
    print(f"Degradation:             {-improvement:.2f} ({-improvement_pct:.1f}%)")

# Add metrics and finalize training results
metrics_to_add = {
    "final_ppl": final_ppl,
    "best_eval_ppl": best_eval_ppl,
    "baseline_ppl": baseline_ppl,
    "relative_improvement": improvement_pct / 100,
    "final_eval_time": final_eval_time,
    "trainable_parameters": trainable,
    "total_parameters": total,
    "training_steps": TRAINING_STEPS
}

for name, value in metrics_to_add.items():
    train_tracker.add_metric(name, float(value))

# Add scale statistics
stats = get_polynomial_stats(model_dendritic)
scales = [abs(s['scale']) for s in stats.values()]
scale_metrics = {
    "avg_scale": sum(scales)/len(scales),
    "max_scale": max(scales),
    "min_scale": min(scales)
}
for name, value in scale_metrics.items():
    train_tracker.add_metric(name, value)

# Save final results
train_results = train_tracker.finalize(model_dendritic)
print(f"\nDendritic training results saved to: results/{train_results['experiment_id']}.json")

# Interpret results
if final_ppl < baseline_ppl * 0.5:
    print("\n✓ Significant improvement! Dendritic layers are learning useful patterns.")
elif final_ppl < baseline_ppl:
    print("\n~ Modest improvement. May need more training or better hyperparameters.")
else:
    print("\n✗ No improvement. Consider different target layers or hyperparameters.")

# Final sample after training (only if we have valid results)
if torch.isfinite(torch.tensor(final_ppl)) and final_ppl != float('inf'):
    sample_prompt = "### Instruction:\nWrite a python function to calculate factorial.\n\n### Input:\n\n\n### Output:\n"
    sample_model_output(model_dendritic, tokenizer, sample_prompt, device)
else:
    print("\nSkipping sample output due to invalid perplexity")

print("\n" + "="*70)
print("SCALE ANALYSIS")
print("="*70)
stats = get_polynomial_stats(model_dendritic)
scales = []
for i, (name, s) in enumerate(stats.items()):
    layer_num = name.split('.')[2] if 'transformer.h.' in name else '?'
    eff_rank_str = f"{s['eff_rank']:.1f}" if s['eff_rank'] is not None else "N/A"
    print(f"Layer {layer_num}: scale={s['scale']:+.6f}, eff_rank={eff_rank_str}/{s['poly_rank']}")
    scales.append(abs(s['scale']))

print(f"\nScale statistics:")
print(f"  Mean |scale|: {sum(scales)/len(scales):.6f}")
print(f"  Max |scale|:  {max(scales):.6f}")
print(f"  Min |scale|:  {min(scales):.6f}")

if max(scales) > 0.1:
    print("  ℹ️ Large scales indicate polynomial pathway is active")
else:
    print("  ⚠️ Small scales suggest polynomial pathway barely contributing")

# 12. Training efficiency analysis
print("\n" + "="*70)
print("TRAINING EFFICIENCY")
print("="*70)
examples_seen = min(TRAINING_STEPS * GRADIENT_ACCUMULATION_STEPS, len(dataset['train']))
print(f"Examples seen:     {examples_seen:,}")
print(f"Dataset size:      {len(dataset['train']):,}")
print(f"Coverage:          {100*examples_seen/len(dataset['train']):.1f}%")
print(f"Param/Example:     {trainable/examples_seen:.1f}")
