# dendritic/experiments/experiment_finetuning.py
"""
Experiment 2: Finetuning Comparison
===================================
Compare finetuning approaches:
- LoRA (standard PEFT method)
- Dendritic enhancement (your method)

Key controls:
- Same pretrained base model
- Same trainable parameter count
- Same training data and hyperparameters
- Multiple random seeds
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from scipy import stats

from transformers.models.gpt2 import GPT2LMHeadModel

# Import peft for LoRA if available
PEFT_AVAILABLE = False
LoraConfig = None
TaskType = None
get_peft_model = None

try:
    from peft import LoraConfig as _LoraConfig, get_peft_model as _get_peft_model, TaskType as _TaskType
    LoraConfig = _LoraConfig
    TaskType = _TaskType
    get_peft_model = _get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    logging.warning("peft not installed. Install with: pip install peft")


@dataclass
class FinetuningConfig:
    """Configuration for finetuning experiment."""
    # Base model
    model_name: str = "gpt2"
    
    # Training
    training_steps: int = 3000
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_length: int = 256
    
    # Dendritic-specific
    target_layers: List[str] = field(default_factory=lambda: ["mlp.c_fc"])
    poly_rank: int = 32
    dendritic_dropout: float = 0.1
    
    # LoRA-specific (rank will be calculated to match params)
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "c_fc", "c_proj"]
    )
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Evaluation
    eval_interval: int = 300
    eval_batches: int = 200
    
    # Experiment
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    output_dir: str = "results/finetuning_comparison"


def get_gpt2_layer_dimensions() -> Dict[str, List[Tuple[int, int]]]:
    """
    Get input/output dimensions for GPT-2 layers.
    
    GPT-2 small (117M):
    - embed_dim = 768
    - num_layers = 12
    - num_heads = 12
    - mlp_hidden = 3072 (4x embed)
    """
    embed_dim = 768
    mlp_hidden = 3072
    
    return {
        # Attention (per layer): c_attn is 768->2304, c_proj is 768->768
        "c_attn": [(embed_dim, 3 * embed_dim)] * 12,  # Q, K, V combined
        "c_proj": [(embed_dim, embed_dim)] * 12,
        # MLP (per layer)
        "c_fc": [(embed_dim, mlp_hidden)] * 12,
        "c_proj_mlp": [(mlp_hidden, embed_dim)] * 12,
    }


def calculate_dendritic_trainable_params(config: FinetuningConfig) -> int:
    """Calculate trainable params when using dendritic enhancement."""
    from .param_utils import count_dendritic_stack_params
    
    layer_dims = get_gpt2_layer_dimensions()
    
    total_params = 0
    for layer_name in config.target_layers:
        if layer_name in layer_dims or layer_name.replace("mlp.", "") in layer_dims:
            # Get the correct key
            key = layer_name if layer_name in layer_dims else layer_name.replace("mlp.", "")
            for in_dim, out_dim in layer_dims.get(key, []):
                # Compute total parameters for the DendriticStack (including bias in both internal layers)
                stack_params = count_dendritic_stack_params(
                    in_dim,
                    out_dim,
                    config.poly_rank,
                    preserve_linear_path=True,
                )
                # Base linear path (frozen) includes weight and bias
                base_linear_params = in_dim * out_dim + out_dim
                # No extra bias subtraction – the base linear bias is already accounted for,
                # and the DendriticStack bias terms are correctly counted in `stack_params`.
                trainable = stack_params - base_linear_params
                total_params += trainable
    
    return total_params


def calculate_matching_lora_rank(
    dendritic_params: int,
    config: FinetuningConfig
) -> int:
    """Calculate LoRA rank to match dendritic trainable params."""
    layer_dims = get_gpt2_layer_dimensions()
    
    # Sum up dimensions for all target modules
    total_dim_sum = 0
    for module_name in config.lora_target_modules:
        if module_name in layer_dims:
            for in_dim, out_dim in layer_dims[module_name]:
                total_dim_sum += in_dim + out_dim
        
        # Special case: 'c_proj' targets both attn and mlp in GPT-2
        if module_name == 'c_proj' and 'c_proj_mlp' in layer_dims:
            for in_dim, out_dim in layer_dims['c_proj_mlp']:
                total_dim_sum += in_dim + out_dim
    
    # LoRA params = rank * total_dim_sum
    rank = dendritic_params / total_dim_sum if total_dim_sum > 0 else 8
    return max(1, round(rank))


def setup_dendritic_model(
    config: FinetuningConfig,
    device: str,
    freeze_linear: bool = True,
) -> GPT2LMHeadModel:
    """Set up GPT-2 with dendritic enhancement."""
    from dendritic.enhancement import enhance_model_with_dendritic
    from dendritic.layers.DendriticStack import DendriticStack
    
    model = GPT2LMHeadModel.from_pretrained(config.model_name)
    
    enhanced_model = enhance_model_with_dendritic(
        model,
        target_layers=config.target_layers,
        poly_rank=config.poly_rank,
        freeze_linear=freeze_linear,
        verbose=False,
        dendritic_cls=DendriticStack,
    )
    
    return enhanced_model.to(device)  # type: ignore


def setup_dendritic_stack_model(
    config: FinetuningConfig,
    device: str
) -> GPT2LMHeadModel:
    """Set up GPT-2 with dendritic stack enhancement."""
    from dendritic.enhancement import enhance_model_with_dendritic
    from dendritic.layers.DendriticStack import DendriticStack
    
    model = GPT2LMHeadModel.from_pretrained(config.model_name)
    
    enhanced_model = enhance_model_with_dendritic(
        model,
        target_layers=config.target_layers,
        poly_rank=config.poly_rank,
        freeze_linear=True,
        verbose=False,
        dendritic_cls=DendriticStack,
        dendritic_kwargs={
            "dropout": config.dendritic_dropout,
        },
    )
    
    return enhanced_model.to(device) # type: ignore


def setup_lora_model(
    config: FinetuningConfig,
    lora_rank: int,
    device: str
) -> Any:
    """Set up GPT-2 with LoRA."""
    if not PEFT_AVAILABLE or LoraConfig is None or TaskType is None or get_peft_model is None:
        raise ImportError("peft library required for LoRA. pip install peft")
    
    model = GPT2LMHeadModel.from_pretrained(config.model_name)
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
    )
    
    peft_model = get_peft_model(model, lora_config)
    
    return peft_model.to(device)


def get_trainable_params(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class FinetuningResult:
    """Results from a single finetuning run."""
    method: str  # "dendritic" or "lora"
    seed: int
    trainable_params: int
    final_eval_loss: float
    final_perplexity: float
    best_eval_loss: float
    best_perplexity: float
    loss_history: List[Dict[str, Any]]
    training_time: float
    baseline_perplexity: float
    polynomial_stats: Optional[Dict[str, Any]] = None


def train_finetuning_run(
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: FinetuningConfig,
    method: str,
    seed: int,
    baseline_ppl: float,
    device: str
) -> FinetuningResult:
    """Train a single finetuning run."""
    import time
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    trainable_params = get_trainable_params(model)
    logging.info(f"{method} trainable params: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Training
    best_eval_loss = float("inf")
    loss_history = []
    train_iter = iter(train_dataloader)
    
    start_time = time.time()
    
    progress = tqdm(range(config.training_steps), desc=f"{method} seed={seed}")
    for step in progress:
        model.train()
        
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
        
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(
            input_ids=inputs,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config.max_grad_norm
        )
        optimizer.step()
        
        progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Evaluation
        if (step + 1) % config.eval_interval == 0:
            eval_loss = evaluate_finetuning(
                model, eval_dataloader, config.eval_batches, device
            )
            perplexity = np.exp(eval_loss)
            
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
            
            loss_history.append({
                "step": step + 1,
                "train_loss": loss.item(),
                "eval_loss": eval_loss,
                "perplexity": perplexity
            })
            
            logging.info(
                f"{method} seed={seed} step={step+1}: "
                f"eval_loss={eval_loss:.4f}, ppl={perplexity:.2f}"
            )
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_eval_loss = evaluate_finetuning(model, eval_dataloader, None, device)
    
    # Get polynomial stats for dendritic layers (for dendritic and dendritic_stack methods)
    if method in ["dendritic", "dendritic_stack"]:
        from dendritic.enhancement import get_polynomial_stats
        polynomial_stats = get_polynomial_stats(model)
    else:
        polynomial_stats = None

    return FinetuningResult(
        method=method,
        seed=seed,
        trainable_params=trainable_params,
        final_eval_loss=final_eval_loss,
        final_perplexity=np.exp(final_eval_loss),
        best_eval_loss=best_eval_loss,
        best_perplexity=np.exp(best_eval_loss),
        loss_history=loss_history,
        training_time=training_time,
        baseline_perplexity=baseline_ppl,
        polynomial_stats=polynomial_stats
    )


def evaluate_finetuning(
    model: nn.Module,
    dataloader: DataLoader,
    max_batches: Optional[int],
    device: str
) -> float:
    """Evaluate finetuned model."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                labels=labels
            )
            
            non_masked = (labels != -100).sum().item()
            if non_masked > 0:
                total_loss += outputs.loss.item() * non_masked
                total_tokens += non_masked
    
    return total_loss / total_tokens if total_tokens > 0 else float("nan")


@dataclass
class FinetuningExperimentResults:
    """Aggregated finetuning experiment results."""
    dendritic_results: List[FinetuningResult]
    dendritic_stack_results: List[FinetuningResult]
    lora_results: List[FinetuningResult]
    statistical_analysis: Dict[str, Any]
    config: FinetuningConfig
    matched_params: Dict[str, int]


def run_finetuning_experiment(
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: Optional[FinetuningConfig] = None,
    device: str = "cuda"
) -> FinetuningExperimentResults:
    """
    Run the full finetuning comparison experiment.
    """
    if config is None:
        config = FinetuningConfig()
    
    if not PEFT_AVAILABLE:
        raise ImportError("peft library required. pip install peft")
    
    logging.info("=" * 70)
    logging.info("FINETUNING EXPERIMENT: Dendritic vs LoRA")
    logging.info("=" * 70)
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate parameter matching
    dendritic_params = calculate_dendritic_trainable_params(config)
    lora_rank = calculate_matching_lora_rank(dendritic_params, config)
    
    logging.info(f"Dendritic trainable params (estimated): {dendritic_params:,}")
    logging.info(f"Matching LoRA rank: {lora_rank}")
    
    # Baseline evaluation
    logging.info("Evaluating baseline (no finetuning)...")
    baseline_model = GPT2LMHeadModel.from_pretrained(config.model_name).to(device) # type: ignore
    baseline_loss = evaluate_finetuning(baseline_model, eval_dataloader, None, device)
    baseline_ppl = np.exp(baseline_loss)
    logging.info(f"Baseline perplexity: {baseline_ppl:.2f}")
    del baseline_model
    torch.cuda.empty_cache()
    
    dendritic_results = []
    dendritic_stack_results = []
    lora_results = []
    
    for seed in config.seeds:
        logging.info(f"\n--- Seed {seed} ---")
        
        # Dendritic finetuning
        logging.info(f"Setting up dendritic model (seed={seed})...")
        dendritic_model = setup_dendritic_model(config, device)
        
        dendritic_result = train_finetuning_run(
            dendritic_model,
            train_dataloader,
            eval_dataloader,
            config,
            "dendritic",
            seed,
            baseline_ppl,
            device
        )
        dendritic_results.append(dendritic_result)
        
        del dendritic_model
        torch.cuda.empty_cache()
        
        # Dendritic Stack finetuning
        logging.info(f"Setting up dendritic stack model (seed={seed})...")
        dendritic_stack_model = setup_dendritic_stack_model(config, device)
        
        dendritic_stack_result = train_finetuning_run(
            dendritic_stack_model,
            train_dataloader,
            eval_dataloader,
            config,
            "dendritic_stack",
            seed,
            baseline_ppl,
            device
        )
        dendritic_stack_results.append(dendritic_stack_result)
        
        del dendritic_stack_model
        torch.cuda.empty_cache()
        
        # LoRA finetuning
        logging.info(f"Setting up LoRA model (seed={seed})...")
        lora_model = setup_lora_model(config, lora_rank, device)
        
        lora_result = train_finetuning_run(
            lora_model,
            train_dataloader,
            eval_dataloader,
            config,
            "lora",
            seed,
            baseline_ppl,
            device
        )
        lora_results.append(lora_result)
        
        del lora_model
        torch.cuda.empty_cache()
    
    # Actual parameter counts
    matched_params = {
        "dendritic": dendritic_results[0].trainable_params if dendritic_results else 0,
        "dendritic_stack": dendritic_stack_results[0].trainable_params if dendritic_stack_results else 0,
        "lora": lora_results[0].trainable_params if lora_results else 0,
        "lora_rank": lora_rank
    }
    
    # Statistical analysis
    statistical_analysis = analyze_finetuning_results(
        dendritic_results, dendritic_stack_results, lora_results, baseline_ppl
    )
    
    results = FinetuningExperimentResults(
        dendritic_results=dendritic_results,
        dendritic_stack_results=dendritic_stack_results,
        lora_results=lora_results,
        statistical_analysis=statistical_analysis,
        config=config,
        matched_params=matched_params
    )
    
    save_finetuning_results(results, output_dir)
    print_finetuning_summary(results)
    
    return results


def analyze_finetuning_results(
    dendritic_results: List[FinetuningResult],
    dendritic_stack_results: List[FinetuningResult],
    lora_results: List[FinetuningResult],
    baseline_ppl: float
) -> Dict[str, Any]:
    """Statistical analysis of finetuning results."""
    
    dendritic_ppl = [r.final_perplexity for r in dendritic_results]
    dendritic_stack_ppl = [r.final_perplexity for r in dendritic_stack_results]
    lora_ppl = [r.final_perplexity for r in lora_results]
    
    dendritic_best_ppl = [r.best_perplexity for r in dendritic_results]
    dendritic_stack_best_ppl = [r.best_perplexity for r in dendritic_stack_results]
    lora_best_ppl = [r.best_perplexity for r in lora_results]
    
    # Paired t-tests
    t_stat_ds, p_value_ds = stats.ttest_rel(dendritic_ppl, dendritic_stack_ppl)
    t_stat_dl, p_value_dl = stats.ttest_rel(dendritic_ppl, lora_ppl)
    t_stat_sl, p_value_sl = stats.ttest_rel(dendritic_stack_ppl, lora_ppl)
    
    # Effect sizes
    diff_ds = np.array(dendritic_ppl) - np.array(dendritic_stack_ppl)
    cohens_ds = np.mean(diff_ds) / np.std(diff_ds, ddof=1) if np.std(diff_ds, ddof=1) > 0 else 0
    
    diff_dl = np.array(dendritic_ppl) - np.array(lora_ppl)
    cohens_dl = np.mean(diff_dl) / np.std(diff_dl, ddof=1) if np.std(diff_dl, ddof=1) > 0 else 0
    
    diff_sl = np.array(dendritic_stack_ppl) - np.array(lora_ppl)
    cohens_sl = np.mean(diff_sl) / np.std(diff_sl, ddof=1) if np.std(diff_sl, ddof=1) > 0 else 0
    
    # Improvement over baseline
    dendritic_improvement = 100 * (baseline_ppl - np.mean(dendritic_ppl)) / baseline_ppl
    dendritic_stack_improvement = 100 * (baseline_ppl - np.mean(dendritic_stack_ppl)) / baseline_ppl
    lora_improvement = 100 * (baseline_ppl - np.mean(lora_ppl)) / baseline_ppl
    
    return {
        "baseline_ppl": baseline_ppl,
        "dendritic": {
            "final_ppl_mean": np.mean(dendritic_ppl),
            "final_ppl_std": np.std(dendritic_ppl, ddof=1),
            "best_ppl_mean": np.mean(dendritic_best_ppl),
            "improvement_over_baseline_pct": dendritic_improvement,
            "individual_ppls": dendritic_ppl
        },
        "dendritic_stack": {
            "final_ppl_mean": np.mean(dendritic_stack_ppl),
            "final_ppl_std": np.std(dendritic_stack_ppl, ddof=1),
            "best_ppl_mean": np.mean(dendritic_stack_best_ppl),
            "improvement_over_baseline_pct": dendritic_stack_improvement,
            "individual_ppls": dendritic_stack_ppl
        },
        "lora": {
            "final_ppl_mean": np.mean(lora_ppl),
            "final_ppl_std": np.std(lora_ppl, ddof=1),
            "best_ppl_mean": np.mean(lora_best_ppl),
            "improvement_over_baseline_pct": lora_improvement,
            "individual_ppls": lora_ppl
        },
        "comparison_dendritic_vs_stack": {
            "ppl_difference": np.mean(dendritic_ppl) - np.mean(dendritic_stack_ppl),
            "stack_better_by_pct": 100 * (np.mean(dendritic_ppl) - np.mean(dendritic_stack_ppl)) / np.mean(dendritic_ppl),
            "paired_ttest": {"t_statistic": t_stat_ds, "p_value": p_value_ds},
            "cohens_d": cohens_ds,
            "significant_005": p_value_ds < 0.05,
            "significant_001": p_value_ds < 0.01
        },
        "comparison_dendritic_vs_lora": {
            "ppl_difference": np.mean(dendritic_ppl) - np.mean(lora_ppl),
            "dendritic_better_by_pct": 100 * (np.mean(lora_ppl) - np.mean(dendritic_ppl)) / np.mean(lora_ppl),
            "paired_ttest": {"t_statistic": t_stat_dl, "p_value": p_value_dl},
            "cohens_d": cohens_dl,
            "significant_005": p_value_dl < 0.05,
            "significant_001": p_value_dl < 0.01
        },
        "comparison_stack_vs_lora": {
            "ppl_difference": np.mean(dendritic_stack_ppl) - np.mean(lora_ppl),
            "stack_better_by_pct": 100 * (np.mean(lora_ppl) - np.mean(dendritic_stack_ppl)) / np.mean(lora_ppl),
            "paired_ttest": {"t_statistic": t_stat_sl, "p_value": p_value_sl},
            "cohens_d": cohens_sl,
            "significant_005": p_value_sl < 0.05,
            "significant_001": p_value_sl < 0.01
        }
    }


def save_finetuning_results(
    results: FinetuningExperimentResults,
    output_dir: Path
) -> None:
    """Save finetuning experiment results after converting numpy types to native types."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def convert_numpy_types(obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    output = {
        "timestamp": timestamp,
        "config": results.config.__dict__,
        "matched_params": results.matched_params,
        "statistical_analysis": convert_numpy_types(results.statistical_analysis),
        "dendritic_runs": [
            {
                "seed": r.seed,
                "trainable_params": r.trainable_params,
                "final_ppl": convert_numpy_types(r.final_perplexity),
                "best_ppl": convert_numpy_types(r.best_perplexity),
                "training_time": r.training_time,
                "loss_history": convert_numpy_types(r.loss_history),
                "polynomial_stats": convert_numpy_types(r.polynomial_stats)
            }
            for r in results.dendritic_results
        ],
        "dendritic_stack_runs": [
            {
                "seed": r.seed,
                "trainable_params": r.trainable_params,
                "final_ppl": convert_numpy_types(r.final_perplexity),
                "best_ppl": convert_numpy_types(r.best_perplexity),
                "training_time": r.training_time,
                "loss_history": convert_numpy_types(r.loss_history),
                "polynomial_stats": convert_numpy_types(r.polynomial_stats)
            }
            for r in results.dendritic_stack_results
        ],
        "lora_runs": [
            {
                "seed": r.seed,
                "trainable_params": r.trainable_params,
                "final_ppl": convert_numpy_types(r.final_perplexity),
                "best_ppl": convert_numpy_types(r.best_perplexity),
                "training_time": r.training_time,
                "loss_history": convert_numpy_types(r.loss_history)
            }
            for r in results.lora_results
        ]
    }
    
    filepath = output_dir / f"finetuning_experiment_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    
    logging.info(f"Results saved to {filepath}")


def print_finetuning_summary(results: FinetuningExperimentResults) -> None:
    """Print formatted finetuning experiment summary."""
    analysis = results.statistical_analysis
    params = results.matched_params
    
    print("\n" + "=" * 70)
    print("FINETUNING EXPERIMENT SUMMARY")
    print("=" * 70)
    
    print(f"\nNumber of seeds: {len(results.config.seeds)}")
    print(f"Training steps: {results.config.training_steps:,}")
    
    print("\n--- Parameter Match ---")
    print(f"Dendritic trainable params:      {params['dendritic']:,}")
    print(f"Dendritic Stack trainable params: {params['dendritic_stack']:,}")
    print(f"LoRA trainable params:           {params['lora']:,}")
    print(f"LoRA rank used:                  {params['lora_rank']}")
    
    print(f"\n--- Baseline ---")
    print(f"Baseline perplexity: {analysis['baseline_ppl']:.2f}")
    
    print("\n--- Final Perplexity ---")
    print(f"Dendritic:       {analysis['dendritic']['final_ppl_mean']:.2f} ± {analysis['dendritic']['final_ppl_std']:.2f}")
    print(f"Dendritic Stack: {analysis['dendritic_stack']['final_ppl_mean']:.2f} ± {analysis['dendritic_stack']['final_ppl_std']:.2f}")
    print(f"LoRA:            {analysis['lora']['final_ppl_mean']:.2f} ± {analysis['lora']['final_ppl_std']:.2f}")
    
    print("\n--- Improvement over Baseline ---")
    print(f"Dendritic:       {analysis['dendritic']['improvement_over_baseline_pct']:.1f}%")
    print(f"Dendritic Stack: {analysis['dendritic_stack']['improvement_over_baseline_pct']:.1f}%")
    print(f"LoRA:            {analysis['lora']['improvement_over_baseline_pct']:.1f}%")
    
    def print_comparison(name: str, comp: Dict[str, Any]) -> None:
        print(f"\n--- {name} Comparison ---")
        print(f"Perplexity difference: {comp['ppl_difference']:.2f}")
        if 'dendritic_better_by_pct' in comp:
            print(f"Dendritic better by: {comp['dendritic_better_by_pct']:.1f}%")
        elif 'stack_better_by_pct' in comp:
            print(f"Stack better by: {comp['stack_better_by_pct']:.1f}%")
        print(f"Paired t-test: t={comp['paired_ttest']['t_statistic']:.3f}, p={comp['paired_ttest']['p_value']:.4f}")
        print(f"Cohen's d: {comp['cohens_d']:.3f}")
        
        # Significance
        if comp['significant_001']:
            print("✓ Result is HIGHLY SIGNIFICANT (p < 0.01)")
        elif comp['significant_005']:
            print("✓ Result is SIGNIFICANT (p < 0.05)")
        else:
            print("✗ Result is NOT statistically significant (p >= 0.05)")
    
    # Print all comparisons
    print_comparison("Dendritic vs Dendritic Stack", analysis['comparison_dendritic_vs_stack'])
    print_comparison("Dendritic vs LoRA", analysis['comparison_dendritic_vs_lora'])
    print_comparison("Dendritic Stack vs LoRA", analysis['comparison_stack_vs_lora'])