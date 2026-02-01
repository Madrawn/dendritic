import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from dendritic.experiments.utils.PretrainingConfig import PretrainingConfig
from dendritic.experiments.utils.experiment_pretraining import PretrainingExperiment


# ----------------------------------------------------------------------
# Minimal dummy dataset – returns random token IDs within vocab range
# ----------------------------------------------------------------------
class DummyDataset(Dataset):
    def __init__(self, config: PretrainingConfig):
        self.config = config

    def __len__(self):
        # Small number of samples – enough for a few training steps
        return 8

    def __getitem__(self, idx):
        # Random token IDs for inputs and labels
        input_ids = torch.randint(0, self.config.vocab_size, (self.config.max_seq_len,))
        labels = torch.randint(0, self.config.vocab_size, (self.config.max_seq_len,))
        return {"input_ids": input_ids, "labels": labels}


# @pytest.mark.slow
@pytest.mark.integration
def test_pretraining_experiment_end_to_end():
    # ------------------------------------------------------------------
    # Configuration – keep everything tiny for a fast test
    # ------------------------------------------------------------------
    config = PretrainingConfig()
    config.training_steps = 5  # Very short training loop
    config.eval_interval = 1  # Evaluate every step
    config.eval_batches = 1  # Only one batch for evaluation
    config.seeds = [0]  # Single seed to keep runtime low
    config.batch_size = 1

    # ------------------------------------------------------------------
    # DataLoaders – use the dummy dataset defined above
    # ------------------------------------------------------------------
    train_dataset = DummyDataset(config)
    eval_dataset = DummyDataset(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # ------------------------------------------------------------------
    # Run the experiment (CPU‑only – CUDA is optional for the test)
    # ------------------------------------------------------------------
    experiment = PretrainingExperiment(config)
    # Create models and define variants (baseline and baseline_wave with AdamW)
    baseline_model, baseline_wave_model = (
        experiment._build_model("standard", dropout=0.0),
        experiment._build_model("standard", dropout=0.0),
    )
    from dendritic.experiments.utils.experiment_pretraining import ModelVariant

    model_variants = [
        ModelVariant(
            name="baseline",
            model=baseline_model,
            results=[],
            optimizer=torch.optim.AdamW(
                baseline_model.parameters(),
                lr=experiment.config.learning_rate,
                weight_decay=experiment.config.weight_decay,
                betas=(0.9, 0.95),
            ),
        ),
        ModelVariant(
            name="baseline_wave",
            model=baseline_wave_model,
            results=[],
            optimizer=torch.optim.AdamW(
                baseline_wave_model.parameters(),
                lr=experiment.config.learning_rate,
                weight_decay=experiment.config.weight_decay,
                betas=(0.9, 0.95),
            ),
        ),
    ]
    results = experiment.run(
        train_loader, eval_loader, model_variants=model_variants, device="cpu"
    )
    # ------------------------------------------------------------------
    # Basic sanity checks on the returned ExperimentResults object
    # ------------------------------------------------------------------
    # The experiment creates two model variants
    # (baseline and baseline_wave) with AdamW optimizer.
    expected_models = {
        "baseline",
        "baseline_wave",
    }
    assert set(results.model_results.keys()) == expected_models

    # Each model should have exactly one TrainingResult (one seed)
    for model_name, result_list in results.model_results.items():
        assert len(result_list) == 1
        result = result_list[0]

        # Verify that the TrainingResult contains the expected fields
        assert isinstance(result.final_train_loss, float)
        assert isinstance(result.final_eval_loss, float)
        assert isinstance(result.final_perplexity, float)
        assert isinstance(result.best_eval_loss, float)
        assert isinstance(result.best_perplexity, float)
        assert isinstance(result.training_time, float)

        # Loss history should contain an entry for each evaluation step
        # (5 steps with eval_interval=1 → 5 entries)
        assert len(result.loss_history) == config.training_steps

        # Polynomial stats dictionary should exist (may be empty for baseline)
        assert hasattr(result, "polynomial_stats")
