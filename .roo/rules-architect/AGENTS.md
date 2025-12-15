# Project Architecture Rules (Non-Obvious Only)

- **Component Dependencies**: The project has a strict dependency flow: dataset_handlers → layers → experiments
- **Model Variants**: The system supports multiple model variants (baseline, dendritic, stack, baseline_wave) that must follow specific interfaces
- **Experiment Structure**: All experiments must follow the pattern in `dendritic/experiments/run_experiments.py`
- **Data Flow**: Data flows through handlers for preprocessing, then through the model, with results collected for analysis
- **Testing Requirements**: All new components must include unit, integration, and edge case tests
- **Performance Considerations**: Memory usage is critical - all experiments must include explicit garbage collection
- **Serialization**: All experiment results must be serializable to JSON with proper type conversion