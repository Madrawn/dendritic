# ConfidenceAwareGPT Experiment Plan

## Objective
Compare training of ConfidenceAwareGPT (with two-pass lookahead training) vs standard MiniGPT (with standard MLP) on language modeling tasks.

## Key Differences from Standard Pretraining

### 1. Model Architecture
- **ConfidenceAwareGPT**: Uses `MetaAwareBlock` with `AdaptiveLayer`, adds `confidence_predictor` head
- **Standard MiniGPT**: Uses standard `TransformerBlock` with `LayerNorm`

### 2. Training Approach
- **ConfidenceAwareGPT**: Two-pass training with lookahead (predicts future loss at t+2)
- **Standard MiniGPT**: Standard next-token prediction

### 3. Data Requirements (Assuming Context Window = L)
- **Standard MiniGPT**: 
  - Needs chunks of length `L + 1`
  - Input: `data[0:L]`, Target: `data[1:L+1]`
  
- **ConfidenceAwareGPT**: 
  - Needs chunks of length `L + 2`
  - Input (`tokens_t`): `data[0:L]`
  - Target 1 (`tokens_t_plus_1`): `data[L]` (Standard Next Token)
  - Target 2 (`tokens_t_plus_2`): `data[L+1]` (Future Consequence Token)

## Implementation Plan

### Phase 1: Configuration & Data Loading

#### 1.1 New Configuration Class
Create `ConfidenceExperimentConfig` extending `PretrainingConfig`:
```python
@dataclass
class ConfidenceExperimentConfig(PretrainingConfig):
    """Configuration for confidence-aware experiments."""
    
    # Confidence-specific parameters
    confidence_alpha: float = 1.0  # Weight for confidence loss
    lookahead_steps: int = 2  # Number of steps to look ahead (fixed at 2 for now)
    confidence_init_bias: float = 2.0  # Initial bias for confidence predictor
    
    # Experiment tracking
    results_dir: str = "results/confidence_experiments"
    save_interval: int = 100  # Save intermediate results every N steps
```

#### 1.2 Modified Data Loading
Create `prepare_confidence_data` function that:
- Loads standard dataset
- Creates sequences of length `seq_len + 2` 
- Returns batches with structure: `(tokens_t, tokens_t_plus_1, tokens_t_plus_2)`
- Maintains compatibility with existing dataset handlers

### Phase 2: Experiment Implementation

#### 2.1 New Experiment Class
Create `ConfidenceAwareExperiment` class:
```python
class ConfidenceAwareExperiment:
    def __init__(self, config: ConfidenceExperimentConfig):
        self.config = config
        self.results_tracker = ConfidenceResultsTracker()
    
    def create_models(self) -> tuple[MiniGPT, ConfidenceAwareGPT]:
        """Create both model variants with parameter matching."""
    
    def train_confidence_model(self, model, train_loader, eval_loader, device):
        """Two-pass training loop using ConfidenceAwareGPT.two_pass_training_step."""
    
    def train_standard_model(self, model, train_loader, eval_loader, device):
        """Standard training loop (reusing existing PretrainingExperiment logic)."""
    
    def run(self) -> ConfidenceExperimentResults:
        """Run full experiment comparing both models."""
```

#### 2.2 Two-Pass Training Loop
Implement training that:
1. For each batch: `(tokens_t, tokens_t_plus_1, tokens_t_plus_2)`
2. Initial confidence: zeros for first step, then previous predictions
3. Call `ConfidenceAwareGPT.two_pass_training_step`
4. Track: `loss_lm`, `loss_confidence`, `total_loss`, `pred_conf_t`

### Phase 3: Results Tracking & Analysis

#### 3.1 Results Data Structure
```python
@dataclass
class ConfidenceTrainingResult(TrainingResult):
    """Extended training results with confidence metrics."""
    confidence_loss_history: list[float]
    token_loss_history: list[float]
    confidence_predictions: list[float]  # Predicted confidence values
    actual_future_losses: list[float]    # Actual future losses for calibration
    
@dataclass  
class ConfidenceExperimentResults:
    standard_model_results: dict[str, list[TrainingResult]]  # By seed
    confidence_model_results: dict[str, list[ConfidenceTrainingResult]]  # By seed
    config: ConfidenceExperimentConfig
    timestamp: str
    training_time: dict[str, float]  # Training time per model type
```

#### 3.2 JSON Serialization
- Save results to `results/confidence_experiments/{timestamp}_results.json`
- Include full configuration, metrics, and training history
- Support loading for analysis and visualization

#### 3.3 Visualization Utilities
Create `plot_confidence_results.py` with functions to:
- Plot loss curves for both models
- Plot confidence vs actual future loss (calibration curve)
- Compare training time and convergence speed
- Generate summary statistics

### Phase 4: Integration & Testing

#### 4.1 Update run_experiments.py
Add new experiment type:
```python
if args.experiment == "confidence":
    logger.info("RUNNING CONFIDENCE-AWARE EXPERIMENT")
    config = ConfidenceExperimentConfig(
        training_steps=1000,  # Reasonable for PoC
        seeds=[42, 123, 456],  # Multiple seeds for statistical significance
        batch_size=4,
        # ... other config
    )
    experiment = ConfidenceAwareExperiment(config)
    results = experiment.run()
```

#### 4.2 Testing Strategy
1. **Unit tests**: Verify two-pass training step logic
2. **Integration tests**: End-to-end experiment with dummy data
3. **Validation**: Compare parameter counts between models
4. **Sanity checks**: Ensure confidence predictions are reasonable

## File Structure

```
dendritic/experiments/confidence/
├── __init__.py
├── config.py                    # ConfidenceExperimentConfig
├── experiment.py               # ConfidenceAwareExperiment class
├── data_loader.py              # Modified data loading for lookahead
├── results.py                  # Results tracking and serialization
├── visualization.py            # Plotting utilities
└── run_confidence_experiment.py # Standalone runner

tests/test_confidence_experiment/
├── test_config.py
├── test_experiment.py
├── test_data_loader.py
└── test_integration.py
```

## Key Challenges & Solutions

### 1. Memory Efficiency
- **Challenge**: Two-pass training requires storing intermediate states
- **Solution**: Use gradient checkpointing and careful memory management

### 2. Training Stability
- **Challenge**: Confidence predictions may become unstable
- **Solution**: Regularization, gradient clipping, careful initialization

### 3. Evaluation Metrics
- **Challenge**: How to evaluate confidence quality
- **Solution**: Use calibration metrics (ECE, reliability diagrams)

### 4. Comparison Fairness
- **Challenge**: Ensure fair comparison (same compute budget)
- **Solution**: Match training steps, not epochs; track wall-clock time

## Success Criteria

1. **Technical**: Experiment runs without errors, produces JSON results
2. **Scientific**: Clear comparison of loss curves and convergence
3. **Usability**: Easy to run with CLI, results easy to analyze
4. **Extensibility**: Framework supports future confidence variants

## Next Steps

1. Implement configuration and data loading modifications
2. Create two-pass training loop
3. Implement results tracking and serialization
4. Integrate into existing experiment framework
5. Test with minimal configuration
6. Run initial PoC experiment
---
  @staticmethod
    def two_pass_training_step(
        model, 
        prev_conf, 
        tokens_t, 
        tokens_t_plus_1, 
        tokens_t_plus_2, 
        alpha=1.0
    ):
        """
        Performs the Lookahead training step.
        
        Assumption: 
        - tokens_t: The input sequence up to time t [B, SeqLen]
        - tokens_t_plus_1: The target for the LM (next token) [B] (scalar integers)
        - tokens_t_plus_2: The target for the future (t+2) [B]
        
        Note: This function calculates loss for the LAST token in the sequence.
        """
        
        # --- Pass 1: Standard LM Training (The "Present") ---
        # We calculate the loss for the CURRENT token prediction
        
        outputs_1 = model(tokens_t, confidence_scalars=prev_conf)
        logits_t = outputs_1["logits"]          # [B, SeqLen, Vocab]
        pred_conf_t = outputs_1["confidence_pred"] # [B, SeqLen]

        # We only care about the prediction at the last step of the sequence
        last_logit = logits_t[:, -1, :]
        last_conf_pred = pred_conf_t[:, -1] # Scalar prediction for the future

        # Standard Cross Entropy for the next token
        loss_lm = F.cross_entropy(last_logit, tokens_t_plus_1)

        # --- Pass 2: Future Consequence Training (The "Lookahead") ---
        with torch.no_grad():
            # 1. Sample the model's actual choice (Hard sampling)
            probs = F.softmax(last_logit, dim=-1)
            # [B, 1]
            predicted_token_id = torch.multinomial(probs, 1).detach()
            
            # 2. Construct the "Hypothetical" sequence
            # Append the PREDICTED token to the input
            # New shape: [B, SeqLen + 1]
            hypothetical_input = torch.cat([tokens_t, predicted_token_id], dim=1)
            
            # Prepare confidence for the next step:
            # We must append the NEW predicted confidence to the history
            # prev_conf: [B, SeqLen, 1]
            # last_conf_pred (reshaped): [B, 1, 1]
            next_step_conf = last_conf_pred.view(-1, 1, 1).detach()
            hypothetical_conf = torch.cat([prev_conf, next_step_conf], dim=1)

        # 3. Run model on hypothetical sequence
        outputs_2 = model(
            hypothetical_input, 
            confidence_scalars=hypothetical_conf
        )
        future_logits = outputs_2["logits"] # [B, SeqLen+1, Vocab]

        # 4. Calculate what the loss WOULD be at t+2
        # We look at the LAST token of this new sequence
        future_logit_step = future_logits[:, -1, :]
        
        # Measure loss against the REAL t+2 token
        loss_future_actual = F.cross_entropy(
            future_logit_step, tokens_t_plus_2, reduction="none"
        )

        # 5. Train the Confidence Head
        # The head at time t (last_conf_pred) should have predicted this future loss
        loss_confidence = F.mse_loss(last_conf_pred, loss_future_actual.detach())

        # Total Backward
        total_loss = loss_lm + (alpha * loss_confidence)
        
        # Note: We return total_loss for logging, but usually you call backward() here 
        # or return it to the optimizer loop.
        # total_loss.backward() 

        return {
            "pred_conf_t": last_conf_pred, 
            "total_loss": total_loss,
            "loss_lm": loss_lm,
            "loss_confidence": loss_confidence
        }