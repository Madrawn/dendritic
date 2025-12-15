# Project Coding Rules (Non-Obvious Only)

- **Dendritic Layer Implementation**: Always use `DendriticLayer` from `dendritic.layers.DendriticLayer` for basic dendritic functionality
- **Polynomial Rank**: When using `poly_rank="auto"`, it computes as `max(4, input_dim // 64)`
- **Model Enhancement**: Use `enhance_model_with_dendritic()` from `dendritic.enhancement` to add dendritic layers to existing models
- **Dataset Handling**: Always use `PythonAlpacaHandler` for dataset processing with specific prompt formatting
- **Test Markers**: Use `@pytest.mark.unit`, `@pytest.mark.integration`, or `@pytest.mark.edge` for test categorization
- **CUDA Requirement**: All tests assume CUDA is available - verify with `assert torch.cuda.is_available()`
- **Memory Management**: Include explicit `gc.collect()` in integration tests due to large model memory requirements
- **Serialization**: Convert numpy types to native Python types for JSON serialization in experiment results