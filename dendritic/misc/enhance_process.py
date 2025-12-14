from typing import cast, Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
import json
from dataclasses import dataclass
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from transformers.generation.utils import GenerationMixin
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import psutil

from dendritic.enhancement import enhance_model_with_dendritic, get_polynomial_stats
from dendritic.layers.DendriticStack import DendriticStack
from dendritic.misc.ExperimentTracker import ExperimentTracker
from dendritic.dataset_handlers.PythonAlpacaHandler import PythonAlpacaHandler

# =====================
# Configuration
# =====================
@dataclass
class TrainingConfig:
    """Configuration for model training."""
    training_steps: int = 6000
    gradient_accumulation_steps: int = 1
    batch_size: int = 4
    eval_interval: int = 300
    max_length: int = 256
    poly_rank: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    target_layers: List[str] | None = None
    dendritic_dropout: float = 0.1

    def __post_init__(self):
        if self.target_layers is None:
            self.target_layers = ["mlp.c_fc"]

# =====================
# Logging Setup
# =====================
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure logging for the application."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    logger = logging.getLogger(__name__)
    return logger

# =====================
# Model Utilities
# =====================
def load_models(device: str) -> Tuple[GPT2LMHeadModel, GPT2LMHeadModel]:
    """Load base and dendritic models."""
    logger = logging.getLogger(__name__)
    logger.info("Loading base models...")
    
    # Load models
    model_base = GPT2LMHeadModel.from_pretrained("gpt2")
    model_dendritic = GPT2LMHeadModel.from_pretrained("gpt2")
    
    # Move models to device
    model_base = model_base.to(device)  # type: ignore
    model_dendritic = model_dendritic.to(device)  # type: ignore
    
    return model_base, model_dendritic

def enhance_model(model: GPT2LMHeadModel, config: TrainingConfig) -> GPT2LMHeadModel:
    """Enhance model with dendritic layers."""
    logger = logging.getLogger(__name__)
    logger.info("Enhancing model with dendritic layers...")
    
    enhanced_model = enhance_model_with_dendritic(
        model,
        target_layers=config.target_layers,
        poly_rank=config.poly_rank,
        freeze_linear=True,
        verbose=True,
        dendritic_cls=DendriticStack,
        dendritic_kwargs={"dropout": config.dendritic_dropout},
    )
    return cast(GPT2LMHeadModel, enhanced_model)

# =====================
# Data Loading
# =====================
def load_and_prepare_data(tokenizer: GPT2Tokenizer, max_length: int) -> Dict[str, Any]:
    """Load and prepare dataset."""
    logger = logging.getLogger(__name__)
    logger.info("Loading and preparing dataset...")
    
    handler = PythonAlpacaHandler(tokenizer, max_length=max_length)
    prepared = handler.prepare_data(test_size=0.1)
    
    logger.info(f"Train size: {len(prepared['train'])}")
    logger.info(f"Test size: {len(prepared['eval'])}")
    
    return prepared

# =====================
# Evaluation
# =====================
def evaluate_model(
    model: GPT2LMHeadModel, 
    dataloader: DataLoader, 
    max_batches: Optional[int] = None,
    device: str = "cuda"
) -> float:
    """Evaluate model perplexity on a dataset."""
    logger = logging.getLogger(__name__)
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    nan_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break

            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            try:
                outputs = model(
                    input_ids=inputs, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
            except Exception as e:
                logger.error(f"Error during evaluation batch {i}: {str(e)}")
                continue

            # Count only non-masked tokens
            non_masked = (labels != -100).sum().item()
            if non_masked == 0:
                logger.warning(f"Batch {i}: all tokens masked, skipping")
                nan_batches += 1
                continue

            if not torch.isfinite(outputs.loss):
                logger.warning(f"Batch {i}: loss is not finite")
                nan_batches += 1
                continue

            total_loss += outputs.loss.item() * non_masked
            total_tokens += non_masked

    if total_tokens == 0:
        logger.error("No valid tokens found for evaluation")
        return float("nan")

    mean_loss = total_loss / total_tokens
    if not torch.isfinite(torch.tensor(mean_loss)):
        logger.error("Mean loss is not finite")
        return float("nan")

    perplexity = torch.exp(torch.tensor(mean_loss)).item()
    
    if nan_batches > 0:
        logger.warning(f"Skipped {nan_batches} batches due to invalid data")

    return perplexity

# =====================
# Training
# =====================
def train_model(
    model: GPT2LMHeadModel,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    tokenizer: GPT2Tokenizer,
    config: TrainingConfig,
    device: str
) -> Dict[str, Any]:
    """Train the model and return training results."""
    logger = logging.getLogger(__name__)
    
    # Setup optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Setup experiment tracking
    tracker = ExperimentTracker(
        method_name="dendritic_finetune",
        params=config.__dict__
    )
    
    # Training loop
    model.train()
    best_eval_ppl = float("inf")
    accumulated_loss = 0.0
    train_iter = iter(train_dataloader)
    
    progress_bar = tqdm(range(config.training_steps), desc="Training", ncols=100)
    for step in progress_bar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        # Prepare batch
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        try:
            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()
        except Exception as e:
            logger.error(f"Error during training step {step}: {str(e)}")
            optimizer.zero_grad()
            continue

        # Gradient accumulation and optimization step
        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                trainable_params, 
                config.max_grad_norm
            )
            optimizer.step()
            optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({
                "avg_loss": f"{accumulated_loss:.4f}"
            })
            accumulated_loss = 0.0

        # Logging and evaluation
        if (step + 1) % 100 == 0:
            log_training_progress(model, step, tracker)
            
        if (step + 1) % config.eval_interval == 0:
            best_eval_ppl = run_evaluation(
                model, 
                eval_dataloader, 
                step, 
                best_eval_ppl, 
                tokenizer, 
                device, 
                tracker
            )
            model.train()

    return {
        "tracker": tracker,
        "best_eval_ppl": best_eval_ppl
    }

def log_training_progress(model: GPT2LMHeadModel, step: int, tracker: ExperimentTracker) -> None:
    """Log training progress and polynomial statistics."""
    stats = get_polynomial_stats(model)
    scales = [s["scale"] for s in stats.values()]
    avg_scale = sum(scales) / len(scales)
    min_scale = min(scales)
    max_scale = max(scales)
    
    logging.info(
        f"Step {step+1:4d}: scale: avg={avg_scale:+.6f}, "
        f"min={min_scale:+.6f}, max={max_scale:+.6f}"
    )
    
    tracker.add_metric("avg_scale", avg_scale, step=step)
    tracker.add_metric("min_scale", min_scale, step=step)
    tracker.add_metric("max_scale", max_scale, step=step)

def run_evaluation(
    model: GPT2LMHeadModel,
    eval_dataloader: DataLoader,
    step: int,
    best_eval_ppl: float,
    tokenizer: GPT2Tokenizer,
    device: str,
    tracker: ExperimentTracker
) -> float:
    """Run evaluation and return best perplexity."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running evaluation at step {step+1}...")
    
    eval_ppl = evaluate_model(model, eval_dataloader, max_batches=200)
    logger.info(f"Evaluation perplexity: {eval_ppl:.2f}")
    
    # Update best perplexity
    if eval_ppl < best_eval_ppl:
        best_eval_ppl = eval_ppl
    
    # Log metrics
    tracker.add_metric("eval_ppl", eval_ppl, step=step)
    tracker.add_metric("best_eval_ppl", best_eval_ppl, step=step)
    
    # Sample model output
    sample_prompt = (
        "### Instruction:\nWrite a python function to calculate factorial.\n\n"
        "### Input:\n\n\n### Output:\n"
    )
    sample_model_output(model, tokenizer, sample_prompt, device)
    
    return best_eval_ppl

# =====================
# Main Function
# =====================
def main():
    """Main training function."""
    # Setup
    logger = setup_logging()
    config = TrainingConfig()
    
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Data
    data = load_and_prepare_data(tokenizer, config.max_length)
    train_dataloader = DataLoader(
        data["train"], 
        batch_size=config.batch_size, 
        shuffle=True
    )
    eval_dataloader = DataLoader(
        data["eval"], 
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    # Models
    model_base, model_dendritic = load_models(device)
    model_dendritic = enhance_model(model_dendritic, config)
    
    # Parameter verification
    trainable = sum(p.numel() for p in model_dendritic.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model_dendritic.parameters())
    logger.info(f"Trainable parameters: {trainable:,} ({100*trainable/total:.2f}%)")
    
    # Quick sanity check
    logger.info("Running quick evaluation...")
    quick_ppl = evaluate_model(model_dendritic, eval_dataloader, max_batches=100)
    logger.info(f"Dendritic (pre-training): {quick_ppl:.2f}")
    
    # Baseline evaluation
    logger.info("Running baseline evaluation...")
    baseline_ppl = evaluate_model(model_base, eval_dataloader)
    logger.info(f"Baseline perplexity: {baseline_ppl:.2f}")
    
    # Training
    logger.info("Starting training...")
    training_results = train_model(
        model_dendritic,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        config,
        device
    )
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_ppl = evaluate_model(model_dendritic, eval_dataloader)
    logger.info(f"Final perplexity: {final_ppl:.2f}")
    
    # Save results
    results = training_results["tracker"].finalize(model_dendritic)
    logger.info(f"Results saved to: results/{results['experiment_id']}.json")
    
    # Print summary
    print_summary(baseline_ppl, final_ppl, training_results["best_eval_ppl"])

def print_summary(baseline_ppl: float, final_ppl: float, best_eval_ppl: float) -> None:
    """Print training summary."""
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print(f"Baseline perplexity:     {baseline_ppl:.2f}")
    print(f"Final perplexity:        {final_ppl:.2f}")
    print(f"Best eval during train:  {best_eval_ppl:.2f}")
    
    improvement = baseline_ppl - final_ppl
    improvement_pct = 100 * improvement / baseline_ppl if baseline_ppl != 0 else 0.0
    
    if improvement > 0:
        print(f"Improvement:             {improvement:.2f} ({improvement_pct:.1f}%)")
    else:
        print(f"Degradation:             {-improvement:.2f} ({-improvement_pct:.1f}%)")

def sample_model_output(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer, 
    prompt: str, 
    device: str, 
    max_new_tokens: int = 64
) -> None:
    """Generate and print sample model output."""
    logger = logging.getLogger(__name__)
    logger.info("Generating sample output...")
    
    try:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        model.eval()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.95,
                temperature=0.8,
            )
        generated = tokenizer.decode(
            output_ids[0][input_ids.shape[-1]:], 
            skip_special_tokens=True
        )
        logger.info(f"Sample output:\n{generated}")
    except Exception as e:
        logger.error(f"Error generating sample: {str(e)}")

if __name__ == "__main__":
    main()
