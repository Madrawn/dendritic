import logging

from experiments.models.MiniGPT import MiniGPT
from .PretrainingConfig import PretrainingConfig
from .param_utils import find_matching_hidden_dims, verify_param_match

def _build_model(
    config: PretrainingConfig,
    hidden_dim: int,
    mlp_type: str,
    dropout: float,
    poly_rank: int | None = None,
) -> MiniGPT:
    """Helper to construct a MiniGPT model with given parameters."""
    return MiniGPT(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        hidden_dim=hidden_dim,
        mlp_type=mlp_type,
        dropout=dropout,
        **({} if poly_rank is None else {"poly_rank": poly_rank}),
    )

def create_models(
    config: PretrainingConfig,
) -> tuple[MiniGPT, MiniGPT, MiniGPT, MiniGPT]:
    """Create baseline, dendritic, and stack models with matched parameters."""
    baseline_hidden, dendritic_hidden, stack_hidden = find_matching_hidden_dims(config)

    logging.info(f"Baseline hidden dim: {baseline_hidden}")
    logging.info(f"Dendritic hidden dim: {dendritic_hidden}")
    logging.info(f"Dendritic Stack hidden dim: {stack_hidden}")

    baseline_model = _build_model(
        config,
        hidden_dim=baseline_hidden,
        mlp_type="baseline",
        dropout=config.dropout,
    )
    baseline_wave_model = _build_model(
        config,
        hidden_dim=baseline_hidden,
        mlp_type="baseline_wave",
        dropout=0,
    )
    dendritic_model = _build_model(
        config,
        hidden_dim=dendritic_hidden,
        mlp_type="dendritic",
        dropout=config.dropout,
        poly_rank=config.poly_rank,
    )
    stack_model = _build_model(
        config,
        hidden_dim=stack_hidden,
        mlp_type="dendritic_stack",
        dropout=config.dropout,
        poly_rank=config.poly_rank,
    )

    # Verify parameter matches
    matched, details = verify_param_match(baseline_model, dendritic_model, tolerance=0.02)
    logging.info(
        f"Baseline vs Dendritic: {matched} (diff: {details['relative_diff']:.2%})"
    )
    matched_stack, details_stack = verify_param_match(
        baseline_model, stack_model, tolerance=0.02
    )
    logging.info(
        f"Baseline vs DendriticStack: {matched_stack} (diff: {details_stack['relative_diff']:.2%})"
    )
    if not matched or not matched_stack:
        logging.warning("Parameters not matched within 2% tolerance!")

    # Store computed hidden dims in config
    config.baseline_hidden_dim = baseline_hidden
    config.dendritic_hidden_dim = dendritic_hidden
    config.dendritic_stack_hidden_dim = stack_hidden

    return baseline_model, dendritic_model, stack_model, baseline_wave_model