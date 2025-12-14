# dendritic/experiments/experiment_pretraining.py
"""
Experiment 1: Pretraining Comparison
====================================
Compare fresh training of:
- Baseline GPT-2 with standard MLP
- Dendritic GPT-2 with DendriticMLP (parameter-matched)

Key controls:
- Same total parameter count
- Same training data
- Same hyperparameters (except architecture-specific)
- Multiple random seeds for statistical significance
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from datetime import datetime
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# For statistical analysis
from scipy import stats


@dataclass
class PretrainingConfig:
    """Configuration for pretraining experiment."""
    # Model architecture
    vocab_size: int = 50257
    embed_dim: int = 384        # Smaller for faster experiments
    num_heads: int = 6
    num_layers: int = 6
    max_seq_len: int = 256
    dropout: float = 0.1
    
    # Dendritic-specific
    poly_rank: int = 16
    dendritic_dropout: float = 0.1
    
    # Training
    training_steps: int = 10000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Evaluation
    eval_interval: int = 500
    eval_batches: int = 100
    
    # Experiment
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 1011])
    output_dir: str = "results/pretraining_comparison"
    
    # Computed fields (set in __post_init__)
    baseline_hidden_dim: int = 0
    dendritic_hidden_dim: int = 0
    dendritic_stack_hidden_dim: int = 0
    
    def __post_init__(self):
        # We'll compute these after calculating non-MLP params
        pass


class BaselineMLP(nn.Module):
    """Standard MLP block for transformer."""
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DendriticPretrainingMLP(nn.Module):
    """
    Dendritic MLP for pretraining comparison.
    
    Uses DendriticLayer for fc1 (input projection) where quadratic
    interactions are most valuable.
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        poly_rank: int = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Import here to avoid circular imports
        from dendritic.layers.DendriticLayer import DendriticLayer
        
        self.fc1 = DendriticLayer(
            embed_dim,
            hidden_dim,
            poly_rank=poly_rank,
            init_scale=0.1
        )
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class DendriticStackPretrainingMLP(nn.Module):
    """
    Dendritic Stack MLP for pretraining comparison.
    
    Uses DendriticStack as the main transformation layer.
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        poly_rank: int = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Import here to avoid circular imports
        from dendritic.layers.DendriticStack import DendriticStack
        
        self.stack = DendriticStack(
            input_dim=embed_dim,
            output_dim=hidden_dim,
            poly_rank=poly_rank,
            preserve_linear_path=True
        )
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stack(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with configurable MLP."""
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_module: nn.Module,
        dropout: float = 0.0
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = mlp_module
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        
        # MLP with residual
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class MiniGPT(nn.Module):
    """
    Minimal GPT for pretraining experiments.
    
    Supports baseline, dendritic, and dendritic_stack MLP variants.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        hidden_dim: int,
        mlp_type: str = "baseline",  # "baseline", "dendritic", or "dendritic_stack"
        poly_rank: int = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            if mlp_type == "baseline":
                mlp = BaselineMLP(embed_dim, hidden_dim, dropout)
            elif mlp_type == "dendritic":
                mlp = DendriticPretrainingMLP(
                    embed_dim, hidden_dim, poly_rank, dropout
                )
            elif mlp_type == "dendritic_stack":
                mlp = DendriticStackPretrainingMLP(
                    embed_dim, hidden_dim, poly_rank, dropout
                )
            else:
                raise ValueError(f"Unknown mlp_type: {mlp_type}")
            
            block = TransformerBlock(embed_dim, num_heads, mlp, dropout)
            self.blocks.append(block)
        
        # Output
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying
        self.head.weight = self.tok_emb.weight
        
        # Causal mask (registered as buffer)
        # Create causal mask and register as buffer to ensure proper device placement
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape
        
        # Embeddings
        tok_emb = self.tok_emb(input_ids)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)
        
        # Causal mask for this sequence length
        mask = self.causal_mask[:T, :T] # type: ignore DO NOT TOUCH!
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        output = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
            output["loss"] = loss
        
        return output


def calculate_non_mlp_params(config: PretrainingConfig) -> int:
    """Calculate parameters outside MLP (embeddings, attention, layer norms)."""
    embed_dim = config.embed_dim
    vocab_size = config.vocab_size
    num_layers = config.num_layers
    num_heads = config.num_heads
    max_seq_len = config.max_seq_len
    
    # Token embeddings (shared with output head)
    tok_emb_params = vocab_size * embed_dim
    
    # Position embeddings
    pos_emb_params = max_seq_len * embed_dim
    
    # Per-layer non-MLP params
    per_layer_params = 0
    
    # LayerNorm x2: 2 * 2 * embed_dim (weight + bias)
    per_layer_params += 4 * embed_dim
    
    # Attention: Q, K, V projections + output projection
    # Q, K, V: 3 * (embed_dim * embed_dim + embed_dim)
    # Out: embed_dim * embed_dim + embed_dim
    attn_params = 4 * (embed_dim * embed_dim + embed_dim)
    per_layer_params += attn_params
    
    # Final layer norm
    final_ln_params = 2 * embed_dim
    
    total = (
        tok_emb_params + 
        pos_emb_params + 
        num_layers * per_layer_params + 
        final_ln_params
    )
    
    return total


def calculate_mlp_params_baseline(embed_dim: int, hidden_dim: int) -> int:
    """Calculate standard MLP parameters."""
    # fc1: embed_dim * hidden_dim + hidden_dim
    # fc2: hidden_dim * embed_dim + embed_dim
    return 2 * embed_dim * hidden_dim + hidden_dim + embed_dim


def calculate_mlp_params_dendritic(
    embed_dim: int,
    hidden_dim: int,
    poly_rank: int,
    diag_rank: Optional[int] = None
) -> int:
    """Calculate DendriticMLP parameters."""
    if diag_rank is None:
        diag_rank = max(4, poly_rank // 4)
    
    # DendriticLayer (fc1)
    from dendritic.experiments.param_utils import count_dendritic_layer_params
    dendritic_fc1 = count_dendritic_layer_params(
        embed_dim, hidden_dim, poly_rank, diag_rank, bias=True
    )
    
    # Standard fc2
    fc2_params = hidden_dim * embed_dim + embed_dim
    
    return dendritic_fc1 + fc2_params


def calculate_mlp_params_dendritic_stack(
    embed_dim: int,
    hidden_dim: int,
    poly_rank: int,
    diag_rank: Optional[int] = None
) -> int:
    """Calculate DendriticStack MLP parameters."""
    if diag_rank is None:
        diag_rank = max(4, poly_rank // 4)
    
    # DendriticStack parameters
    from dendritic.experiments.param_utils import count_dendritic_layer_params
    from dendritic.layers.DendriticStack import DendriticStack
    
    # Calculate parameters for the stack
    # Layer1: embed_dim -> bottleneck_dim (poly_rank*2)
    bottleneck_dim = poly_rank * 2
    layer1_params = count_dendritic_layer_params(
        embed_dim, bottleneck_dim, poly_rank, diag_rank, bias=True
    )
    
    # Layer2: bottleneck_dim -> hidden_dim
    layer2_params = count_dendritic_layer_params(
        bottleneck_dim, hidden_dim, poly_rank, diag_rank, bias=True
    )
    
    # Base linear: embed_dim -> hidden_dim
    base_linear_params = embed_dim * hidden_dim + hidden_dim
    
    # Stack total
    stack_params = layer1_params + layer2_params + base_linear_params
    
    # Standard fc2
    fc2_params = hidden_dim * embed_dim + embed_dim
    
    return stack_params + fc2_params


def find_matching_hidden_dims(config: PretrainingConfig) -> Tuple[int, int, int]:
    """
    Find hidden dimensions that give equal total parameters for all three variants.
    
    Returns:
        (baseline_hidden_dim, dendritic_hidden_dim, stack_hidden_dim)
    """
    embed_dim = config.embed_dim
    num_layers = config.num_layers
    poly_rank = config.poly_rank
    
    non_mlp_params = calculate_non_mlp_params(config)
    
    # Standard ratio is 4x embed_dim
    baseline_hidden = 4 * embed_dim
    baseline_mlp_per_layer = calculate_mlp_params_baseline(embed_dim, baseline_hidden)
    baseline_total = non_mlp_params + num_layers * baseline_mlp_per_layer
    
    # Target total params
    target_total = baseline_total
    target_mlp_budget = target_total - non_mlp_params
    target_per_layer = target_mlp_budget / num_layers
    
    # Binary search for dendritic hidden_dim
    diag_rank = max(4, poly_rank // 4)
    
    def dendritic_mlp_params(h: int) -> int:
        return calculate_mlp_params_dendritic(embed_dim, h, poly_rank, diag_rank)
    
    def stack_mlp_params(h: int) -> int:
        return calculate_mlp_params_dendritic_stack(embed_dim, h, poly_rank, diag_rank)
    
    # Search for dendritic hidden_dim
    lo, hi = embed_dim, 8 * embed_dim
    while lo < hi:
        mid = (lo + hi) // 2
        params = dendritic_mlp_params(mid)
        if params < target_per_layer:
            lo = mid + 1
        else:
            hi = mid
    dendritic_hidden = lo
    
    # Fine-tune dendritic hidden_dim
    best_dendritic = dendritic_hidden
    best_diff = abs(dendritic_mlp_params(dendritic_hidden) - target_per_layer)
    for h in [dendritic_hidden - 1, dendritic_hidden, dendritic_hidden + 1]:
        if h > 0:
            diff = abs(dendritic_mlp_params(h) - target_per_layer)
            if diff < best_diff:
                best_diff = diff
                best_dendritic = h
    
    # Search for stack hidden_dim
    lo, hi = embed_dim, 8 * embed_dim
    while lo < hi:
        mid = (lo + hi) // 2
        params = stack_mlp_params(mid)
        if params < target_per_layer:
            lo = mid + 1
        else:
            hi = mid
    stack_hidden = lo
    
    # Fine-tune stack hidden_dim
    best_stack = stack_hidden
    best_diff = abs(stack_mlp_params(stack_hidden) - target_per_layer)
    for h in [stack_hidden - 1, stack_hidden, stack_hidden + 1]:
        if h > 0:
            diff = abs(stack_mlp_params(h) - target_per_layer)
            if diff < best_diff:
                best_diff = diff
                best_stack = h
    
    return baseline_hidden, best_dendritic, best_stack


def create_models(config: PretrainingConfig) -> Tuple[MiniGPT, MiniGPT, MiniGPT]:
    """Create baseline, dendritic, and stack models with matched parameters."""
    baseline_hidden, dendritic_hidden, stack_hidden = find_matching_hidden_dims(config)
    
    logging.info(f"Baseline hidden dim: {baseline_hidden}")
    logging.info(f"Dendritic hidden dim: {dendritic_hidden}")
    logging.info(f"Dendritic Stack hidden dim: {stack_hidden}")
    
    baseline_model = MiniGPT(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        hidden_dim=baseline_hidden,
        mlp_type="baseline",
        dropout=config.dropout
    )
    
    dendritic_model = MiniGPT(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        hidden_dim=dendritic_hidden,
        mlp_type="dendritic",
        poly_rank=config.poly_rank,
        dropout=config.dropout
    )
    
    stack_model = MiniGPT(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        hidden_dim=stack_hidden,
        mlp_type="dendritic_stack",
        poly_rank=config.poly_rank,
        dropout=config.dropout
    )
    
    # Verify parameter matches
    from dendritic.experiments.param_utils import verify_param_match
    
    # Verify baseline vs dendritic
    matched, details = verify_param_match(baseline_model, dendritic_model, tolerance=0.02)
    logging.info(f"Baseline vs Dendritic: {matched} (diff: {details['relative_diff']:.2%})")
    
    # Verify baseline vs stack
    matched_stack, details_stack = verify_param_match(baseline_model, stack_model, tolerance=0.02)
    logging.info(f"Baseline vs DendriticStack: {matched_stack} (diff: {details_stack['relative_diff']:.2%})")
    
    if not matched or not matched_stack:
        logging.warning("Parameters not matched within 2% tolerance!")
    
    # Store computed hidden dims in config
    config.baseline_hidden_dim = baseline_hidden
    config.dendritic_hidden_dim = dendritic_hidden
    config.dendritic_stack_hidden_dim = stack_hidden
    
    return baseline_model, dendritic_model, stack_model


@dataclass
class TrainingResult:
    """Results from a single training run."""
    model_type: str
    seed: int
    final_train_loss: float
    final_eval_loss: float
    final_perplexity: float
    best_eval_loss: float
    best_perplexity: float
    loss_history: List[Dict[str, Any]]
    training_time: float
    config: Dict[str, Any]
    polynomial_stats: Dict[str, Any] = field(default_factory=dict)


def train_single_run(
    model: MiniGPT,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: PretrainingConfig,
    model_type: str,
    seed: int,
    device: str
) -> TrainingResult:
    """Train a single model and return results."""
    import time
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model.to(device)
    
    # Optimizer with warmup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Linear warmup then cosine decay
    def lr_lambda(step: int) -> float:
        if step < config.warmup_steps:
            return step / config.warmup_steps
        progress = (step - config.warmup_steps) / (config.training_steps - config.warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training state
    best_eval_loss = float("inf")
    loss_history = []
    train_iter = iter(train_dataloader)
    
    start_time = time.time()
    loss = None
    progress = tqdm(range(config.training_steps), desc=f"{model_type} seed={seed}")
    for step in progress:
        model.train()
        
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        # Update progress
        progress.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Evaluation
        if (step + 1) % config.eval_interval == 0:
            eval_loss = evaluate(model, eval_dataloader, config.eval_batches, device)
            perplexity = np.exp(eval_loss)
            
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
            
            loss_history.append({
                "step": step + 1,
                "train_loss": loss.item(),
                "eval_loss": eval_loss,
                "perplexity": perplexity,
                "lr": scheduler.get_last_lr()[0]
            })
            
            logging.info(
                f"{model_type} seed={seed} step={step+1}: "
                f"train_loss={loss.item():.4f}, eval_loss={eval_loss:.4f}, "
                f"ppl={perplexity:.2f}"
            )
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_eval_loss = evaluate(model, eval_dataloader, None, device)  # Full eval
    
    # Get polynomial stats for dendritic layers
    from dendritic.enhancement import get_polynomial_stats
    polynomial_stats = get_polynomial_stats(model)
    

    # Use last loss if available, else use final_eval_loss
    final_train_loss = loss.item() if loss is not None else final_eval_loss

    return TrainingResult(
        model_type=model_type,
        seed=seed,
        final_train_loss=final_train_loss,
        final_eval_loss=final_eval_loss,
        final_perplexity=float(np.exp(final_eval_loss)),
        best_eval_loss=best_eval_loss,
        best_perplexity=float(np.exp(best_eval_loss)),
        loss_history=loss_history,
        training_time=training_time,
        config=config.__dict__,
        polynomial_stats=polynomial_stats
    )


def evaluate(
    model: MiniGPT,
    dataloader: DataLoader,
    max_batches: Optional[int],
    device: str
) -> float:
    """Evaluate model and return mean loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, labels=labels)
            
            # Count non-masked tokens
            non_masked = (labels != -100).sum().item()
            total_loss += outputs["loss"].item() * non_masked
            total_tokens += non_masked
    
    return total_loss / total_tokens if total_tokens > 0 else float("nan")


@dataclass
class ExperimentResults:
    """Aggregated results from pretraining experiment."""
    baseline_results: List[TrainingResult]
    dendritic_results: List[TrainingResult]
    stack_results: List[TrainingResult]
    statistical_analysis: Dict[str, Any]
    config: PretrainingConfig


def run_pretraining_experiment(
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: Optional[PretrainingConfig] = None,
    device: str = "cuda"
) -> ExperimentResults:
    """
    Run the full pretraining comparison experiment.
    
    Args:
        train_dataloader: Training data
        eval_dataloader: Evaluation data
        config: Experiment configuration
        device: Device to train on
        
    Returns:
        ExperimentResults with statistics
    """
    if config is None:
        config = PretrainingConfig()
    
    logging.info("=" * 70)
    logging.info("PRETRAINING EXPERIMENT")
    logging.info("=" * 70)
    
    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_results = []
    dendritic_results = []
    stack_results = []
    
    for seed in config.seeds:
        logging.info(f"\n--- Seed {seed} ---")
        
        # Create fresh models for each seed
        baseline_model, dendritic_model, stack_model = create_models(config)
        
        # Train baseline
        logging.info(f"Training baseline (seed={seed})...")
        baseline_result = train_single_run(
            baseline_model,
            train_dataloader,
            eval_dataloader,
            config,
            "baseline",
            seed,
            device
        )
        baseline_results.append(baseline_result)
        
        # Train dendritic
        logging.info(f"Training dendritic (seed={seed})...")
        dendritic_result = train_single_run(
            dendritic_model,
            train_dataloader,
            eval_dataloader,
            config,
            "dendritic",
            seed,
            device
        )
        dendritic_results.append(dendritic_result)
        
        # Train dendritic stack
        logging.info(f"Training dendritic stack (seed={seed})...")
        stack_result = train_single_run(
            stack_model,
            train_dataloader,
            eval_dataloader,
            config,
            "dendritic_stack",
            seed,
            device
        )
        stack_results.append(stack_result)
        
        # Clear GPU memory
        del baseline_model, dendritic_model, stack_model
        torch.cuda.empty_cache()
    
    # Statistical analysis
    statistical_analysis = analyze_results(baseline_results, dendritic_results, stack_results)
    
    # Create results object
    results = ExperimentResults(
        baseline_results=baseline_results,
        dendritic_results=dendritic_results,
        stack_results=stack_results,
        statistical_analysis=statistical_analysis,
        config=config
    )
    
    # Save results
    save_experiment_results(results, output_dir)
    
    # Print summary
    print_experiment_summary(results)
    
    return results


def analyze_results(
    baseline_results: List[TrainingResult],
    dendritic_results: List[TrainingResult],
    stack_results: List[TrainingResult]
) -> Dict[str, Any]:
    """Perform statistical analysis on experiment results."""
    
    # Extract final perplexities
    baseline_ppl = [r.final_perplexity for r in baseline_results]
    dendritic_ppl = [r.final_perplexity for r in dendritic_results]
    stack_ppl = [r.final_perplexity for r in stack_results]
    
    # Best perplexities during training
    baseline_best_ppl = [r.best_perplexity for r in baseline_results]
    dendritic_best_ppl = [r.best_perplexity for r in dendritic_results]
    stack_best_ppl = [r.best_perplexity for r in stack_results]
    
    # Paired t-tests
    t_stat_bd, p_value_bd = stats.ttest_rel(baseline_ppl, dendritic_ppl)
    t_stat_bs, p_value_bs = stats.ttest_rel(baseline_ppl, stack_ppl)
    t_stat_ds, p_value_ds = stats.ttest_rel(dendritic_ppl, stack_ppl)
    
    # Effect sizes
    diff_bd = np.array(baseline_ppl) - np.array(dendritic_ppl)
    cohens_d_bd = np.mean(diff_bd) / np.std(diff_bd, ddof=1)
    
    diff_bs = np.array(baseline_ppl) - np.array(stack_ppl)
    cohens_d_bs = np.mean(diff_bs) / np.std(diff_bs, ddof=1)
    
    diff_ds = np.array(dendritic_ppl) - np.array(stack_ppl)
    cohens_d_ds = np.mean(diff_ds) / np.std(diff_ds, ddof=1)
    
    return {
        "baseline": {
            "final_ppl_mean": np.mean(baseline_ppl),
            "final_ppl_std": np.std(baseline_ppl, ddof=1),
            "best_ppl_mean": np.mean(baseline_best_ppl),
            "best_ppl_std": np.std(baseline_best_ppl, ddof=1),
            "individual_ppls": baseline_ppl
        },
        "dendritic": {
            "final_ppl_mean": np.mean(dendritic_ppl),
            "final_ppl_std": np.std(dendritic_ppl, ddof=1),
            "best_ppl_mean": np.mean(dendritic_best_ppl),
            "best_ppl_std": np.std(dendritic_best_ppl, ddof=1),
            "individual_ppls": dendritic_ppl
        },
        "stack": {
            "final_ppl_mean": np.mean(stack_ppl),
            "final_ppl_std": np.std(stack_ppl, ddof=1),
            "best_ppl_mean": np.mean(stack_best_ppl),
            "best_ppl_std": np.std(stack_best_ppl, ddof=1),
            "individual_ppls": stack_ppl
        },
        "comparison_baseline_dendritic": {
            "ppl_difference": np.mean(baseline_ppl) - np.mean(dendritic_ppl),
            "ppl_improvement_pct": 100 * (np.mean(baseline_ppl) - np.mean(dendritic_ppl)) / np.mean(baseline_ppl),
            "paired_ttest": {"t_statistic": t_stat_bd, "p_value": p_value_bd},
            "cohens_d": cohens_d_bd,
            "significant_005": p_value_bd < 0.05,
            "significant_001": p_value_bd < 0.01
        },
        "comparison_baseline_stack": {
            "ppl_difference": np.mean(baseline_ppl) - np.mean(stack_ppl),
            "ppl_improvement_pct": 100 * (np.mean(baseline_ppl) - np.mean(stack_ppl)) / np.mean(baseline_ppl),
            "paired_ttest": {"t_statistic": t_stat_bs, "p_value": p_value_bs},
            "cohens_d": cohens_d_bs,
            "significant_005": p_value_bs < 0.05,
            "significant_001": p_value_bs < 0.01
        },
        "comparison_dendritic_stack": {
            "ppl_difference": np.mean(dendritic_ppl) - np.mean(stack_ppl),
            "ppl_improvement_pct": 100 * (np.mean(dendritic_ppl) - np.mean(stack_ppl)) / np.mean(dendritic_ppl),
            "paired_ttest": {"t_statistic": t_stat_ds, "p_value": p_value_ds},
            "cohens_d": cohens_d_ds,
            "significant_005": p_value_ds < 0.05,
            "significant_001": p_value_ds < 0.01
        }
    }


def save_experiment_results(results: ExperimentResults, output_dir: Path) -> None:
    """Save experiment results to JSON after converting numpy types to native types."""
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
    
    # Convert to serializable format
    output = {
        "timestamp": timestamp,
        "config": results.config.__dict__,
        "statistical_analysis": convert_numpy_types(results.statistical_analysis),
        "baseline_runs": [
            {
                "seed": r.seed,
                "final_ppl": convert_numpy_types(r.final_perplexity),
                "best_ppl": convert_numpy_types(r.best_perplexity),
                "training_time": r.training_time,
                "loss_history": convert_numpy_types(r.loss_history),
                "polynomial_stats": convert_numpy_types(r.polynomial_stats)
            }
            for r in results.baseline_results
        ],
        "dendritic_runs": [
            {
                "seed": r.seed,
                "final_ppl": convert_numpy_types(r.final_perplexity),
                "best_ppl": convert_numpy_types(r.best_perplexity),
                "training_time": r.training_time,
                "loss_history": convert_numpy_types(r.loss_history),
                "polynomial_stats": convert_numpy_types(r.polynomial_stats)
            }
            for r in results.dendritic_results
        ],
        "stack_runs": [
            {
                "seed": r.seed,
                "final_ppl": convert_numpy_types(r.final_perplexity),
                "best_ppl": convert_numpy_types(r.best_perplexity),
                "training_time": r.training_time,
                "loss_history": convert_numpy_types(r.loss_history),
                "polynomial_stats": convert_numpy_types(r.polynomial_stats)
            }
            for r in results.stack_results
        ]
    }
    
    filepath = output_dir / f"pretraining_experiment_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    
    logging.info(f"Results saved to {filepath}")


def print_experiment_summary(results: ExperimentResults) -> None:
    """Print formatted experiment summary with all three model types."""
    analysis = results.statistical_analysis
    
    print("\n" + "=" * 70)
    print("PRETRAINING EXPERIMENT SUMMARY")
    print("=" * 70)
    
    # Model configuration
    print("\n--- Model Configuration ---")
    print(f"Number of seeds: {len(results.config.seeds)}")
    print(f"Training steps: {results.config.training_steps:,}")
    print(f"Baseline hidden dim:        {results.config.baseline_hidden_dim}")
    print(f"Dendritic hidden dim:       {results.config.dendritic_hidden_dim}")
    print(f"Dendritic Stack hidden dim: {results.config.dendritic_stack_hidden_dim}")
    
    # Final perplexity results
    print("\n--- Final Perplexity (mean ± std) ---")
    print(f"Baseline:        {analysis['baseline']['final_ppl_mean']:8.2f} ± {analysis['baseline']['final_ppl_std']:5.2f}")
    print(f"Dendritic:       {analysis['dendritic']['final_ppl_mean']:8.2f} ± {analysis['dendritic']['final_ppl_std']:5.2f}")
    print(f"Dendritic Stack: {analysis['stack']['final_ppl_mean']:8.2f} ± {analysis['stack']['final_ppl_std']:5.2f}")
    
    # Best perplexity results
    print("\n--- Best Perplexity (mean ± std) ---")
    print(f"Baseline:        {analysis['baseline']['best_ppl_mean']:8.2f} ± {analysis['baseline']['best_ppl_std']:5.2f}")
    print(f"Dendritic:       {analysis['dendritic']['best_ppl_mean']:8.2f} ± {analysis['dendritic']['best_ppl_std']:5.2f}")
    print(f"Dendritic Stack: {analysis['stack']['best_ppl_mean']:8.2f} ± {analysis['stack']['best_ppl_std']:5.2f}")
    
    # Print comparisons
    def print_comparison(name, comp):
        print(f"\n--- {name} Comparison ---")
        print(f"Perplexity difference: {comp['ppl_difference']:7.2f}")
        print(f"Improvement:           {comp['ppl_improvement_pct']:6.1f}%")
        print(f"Paired t-test: t = {comp['paired_ttest']['t_statistic']:6.3f}, p = {comp['paired_ttest']['p_value']:6.4f}")
        print(f"Cohen's d:            {comp['cohens_d']:7.3f}")
        
        # Significance
        if comp['significant_001']:
            sig = "HIGHLY SIGNIFICANT (p < 0.01)"
        elif comp['significant_005']:
            sig = "SIGNIFICANT (p < 0.05)"
        else:
            sig = "NOT significant (p >= 0.05)"
        print(f"Significance:          {sig}")
        
        # Effect size
        d = abs(comp['cohens_d'])
        if d < 0.2:
            effect = "negligible"
        elif d < 0.5:
            effect = "small"
        elif d < 0.8:
            effect = "medium"
        else:
            effect = "large"
        print(f"Effect size:           {effect} (|d| = {d:.2f})")
    
    # All pairwise comparisons
    print_comparison("Baseline vs Dendritic", analysis['comparison_baseline_dendritic'])
    print_comparison("Baseline vs Dendritic Stack", analysis['comparison_baseline_stack'])
    print_comparison("Dendritic vs Dendritic Stack", analysis['comparison_dendritic_stack'])