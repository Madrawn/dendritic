# Project Documentation Rules (Non-Obvious Only)

- **Code Organization**: The project follows a modular structure with clear separation between layers, experiments, and dataset handlers
- **Key Components**: Main components are in `dendritic/layers/` (neural network layers), `dendritic/experiments/` (training and evaluation), and `dendritic/dataset_handlers/` (data processing)
- **Experiment Flow**: Experiments follow a specific pattern: configuration → data loading → model training → analysis → visualization
- **Model Variants**: The project includes several model variants: baseline, dendritic, stack, and baseline_wave
- **Testing Structure**: Tests are organized by type (unit, integration, edge) and use specific pytest markers