
# @pytest.mark.timeout(60)
# @pytest.mark.integration
# def test_pretraining_experiment_wikitext_end_to_end():
#     """
#     End-to-end pretraining test using the real WikiText dataset.

#     The test runs a very short training loop (5 steps) with a batch size of 5
#     to verify that the full data pipeline (handler → tokenisation →
#     grouping → DataLoader) works without requiring large compute resources.
#     """

#     # ------------------------------------------------------------------
#     # Configuration – tiny for a fast test
#     # ------------------------------------------------------------------
#     config = PretrainingConfig()
#     config.training_steps = 5          # Very short training loop
#     config.eval_interval = 1          # Evaluate every step
#     config.eval_batches = 1           # One eval batch
#     config.seeds = [0]                # Single seed for determinism
#     config.batch_size = 5             # Small batch size for the test
#     config.dataset = "wikitext"       # Use the WikiText handler
#     config.dataset_kwargs = {}        # No extra kwargs needed

#     # ------------------------------------------------------------------
#     # Tokenizer – use the same tokenizer the project expects
#     # ------------------------------------------------------------------
#     tokenizer = AutoTokenizer.from_pretrained("gpt2")
#     tokenizer.pad_token = tokenizer.eos_token  # Ensure a pad token exists

#     # ------------------------------------------------------------------
#     # Load the WikiText data via the factory
#     # ------------------------------------------------------------------
#     handler = get_handler(
#         name=config.dataset,
#         tokenizer=tokenizer,
#         max_length=config.max_seq_len,
#         **config.dataset_kwargs,
#     )
#     # Prepare the pretraining DataLoaders (train + eval)
#     dataloaders = handler.prepare_pretraining_data(
#         config=config,
#         num_workers=0,          # Keep worker count low for CI environments
#         streaming=False,        # Load full dataset (small enough for test)
#     )
#     train_loader = dataloaders["train"]
#     eval_loader = dataloaders["eval"]

#     # ------------------------------------------------------------------
#     # Run the experiment (CPU‑only)
#     # ------------------------------------------------------------------
#     experiment = PretrainingExperiment(config)
#     baseline_model, dendritic_model, stack_model, baseline_wave_model = experiment.create_models()
#     from dendritic.experiments.utils.experiment_pretraining import ModelVariant

#     model_variants = [
#         ModelVariant(
#             name="baseline",
#             model=baseline_model,
#             results=[],
#             optimizer=torch.optim.AdamW(
#                 baseline_model.parameters(),
#                 lr=experiment.config.learning_rate,
#                 weight_decay=experiment.config.weight_decay,
#                 betas=(0.9, 0.95),
#             ),
#         ),
#         ModelVariant(
#             name="baseline_wave",
#             model=baseline_wave_model,
#             results=[],
#             optimizer=torch.optim.AdamW(
#                 baseline_wave_model.parameters(),
#                 lr=experiment.config.learning_rate,
#                 weight_decay=experiment.config.weight_decay,
#                 betas=(0.9, 0.95),
#             ),
#         ),
#     ]

#     results = experiment.run(
#         train_loader,
#         eval_loader,
#         model_variants=model_variants,
#         device="cpu",
#     )

#     # ------------------------------------------------------------------
#     # Basic sanity checks
#     # ------------------------------------------------------------------
#     expected_models = {"baseline", "baseline_wave"}
#     assert set(results.model_results.keys()) == expected_models

#     for model_name, result_list in results.model_results.items():
#         # One seed → one TrainingResult per model
#         assert len(result_list) == 1
#         result = result_list[0]

#         # Verify expected numeric fields are present
#         assert isinstance(result.final_train_loss, float)
#         assert isinstance(result.final_eval_loss, float)
#         assert isinstance(result.final_perplexity, float)
#         assert isinstance(result.best_eval_loss, float)
#         assert isinstance(result.best_perplexity, float)
#         assert isinstance(result.training_time, float)

#         # Loss history length should match the number of evaluation steps
#         assert len(result.loss_history) == config.training_steps

#         # Polynomial stats attribute should exist (may be empty for baseline)
#         assert hasattr(result, "polynomial_stats")