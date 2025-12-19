# Dataset Refactoring Plan

## Objective
Enable easy swapping of datasets for pretraining (and eventually finetuning) experiments, starting with OpenWebMath, while maintaining backward compatibility and modular design.

## Current Progress

### Completed

1. **Abstract base class `TextCorpusHandler`** (located at `dendritic/dataset_handlers/TextCorpusHandler.py`) has been updated with the following design decisions:
   - `max_samples` is now a **mandatory** parameter for `load_default_data`. There is no default; callers must specify how many samples they want.
   - `streaming` defaults to `True`. This ensures we never accidentally download huge datasets upfront.
   - **Predefined splits have been removed.** All datasets are treated as a single split (typically “train”) and are split in‑memory using `Dataset.train_test_split`. This drastically simplifies the logic and matches the user’s requirement: “split should always happen in memory by doing it like `Dataset.from_list(list(dataset_head)).train_test_split(test_size=0.1)`”.
   - The loading pipeline now always converts a streaming dataset to a regular `Dataset` after taking `max_samples`, then splits it. Non‑streaming datasets are simply limited to `max_samples` with `.select()` before splitting.

2. **Concrete handler `OpenWebMathHandler`** ( `dendritic/dataset_handlers/OpenWebMathHandler.py` ) has been simplified:
   - Removed the `predefined_splits` attribute and corresponding constructor parameter.
   - No longer overrides `load_default_data` (relies on the parent implementation).

3. **Concrete handler `WikiTextHandler`** ( `dendritic/dataset_handlers/WikiTextHandler.py` ) has been updated:
   - Removed `predefined_splits` attribute and constructor parameter.
   - Kept its override of `load_default_data` only to inject the sub‑dataset name `"wikitext-103-raw-v1"`. The override now respects the new signature (mandatory `max_samples`, default `streaming=True`).

4. **Factory registration** ( `dendritic/dataset_handlers/factory.py` ) remains unchanged; both handlers are auto‑registered under the names `"wikitext"` and `"openwebmath"`.

### What Still Works

- The existing `PretrainingConfig` already contains a `dataset` field (added earlier). The CLI argument `--dataset` is already supported by `run_experiments.py`.
- The `BaseDatasetHandler` abstract interface is unchanged, preserving compatibility with `PythonAlpacaHandler` and any other future handlers.
- The `load_pretraining_data` function still uses the factory to obtain a handler; it only needs to be updated to pass the new mandatory `max_samples` parameter (see Outstanding Tasks).

## Outstanding Tasks

### 1. Update `load_pretraining_data` in `run_experiments.py`
   - Compute `total_raw_samples = num_train_samples + num_eval_samples` (the same numbers already calculated for the old selection logic).
   - Replace the call `handler.load_default_data(test_size=0.0)` with:
     ```python
     raw = handler.load_default_data(
         max_samples=total_raw_samples,
         split="train",
         test_size=num_eval_samples / total_raw_samples,
         streaming=True,
         seed=42,
     )
     ```
   - Keep the subsequent filtering (empty‑line removal) and safe‑selection logic unchanged; it will now operate on the already‑split `train_raw` and `eval_raw` Datasets.
   - Verify that the function still works with both wikitext and openwebmath.

### 2. Update Tests
   - The existing test `test_openwebmath_final.py` (and similar) should be updated to reflect the new mandatory `max_samples` parameter and the changed behavior (streaming default, in‑memory split).
   - Ensure that unit and integration tests for dataset handlers still pass.
   - Run the full test suite with `python run_tests.py` to catch any regressions.

### 3. Validate End‑to‑End
   - Run a small‑scale pretraining experiment with OpenWebMath to confirm the pipeline works correctly:
     ```bash
     python -m dendritic.experiments.run_experiments --experiment pretraining --dataset openwebmath --log-level INFO
     ```
   - Do the same with wikitext to ensure backward compatibility.

### 4. Documentation
   - Update `AGENTS.md` with the new dataset‑addition pattern (emphasize mandatory `max_samples`, default streaming, and the removal of predefined splits).
   - Add a short note about the simplified splitting approach.

## Design Rationale

- **Mandatory `max_samples`**: There is no legitimate use case where we want to load an unbounded dataset; experiments always have a known compute budget (steps × batch size). Requiring an explicit limit prevents accidental downloads of multi‑gigabyte corpora.
- **Streaming default**: Downloads only the requested number of samples, which is essential for large datasets like OpenWebMath. Users can still opt‑out with `streaming=False` if they want to cache the whole dataset locally.
- **Removal of predefined splits**: The previous generic support for predefined splits introduced considerable complexity and corner cases (e.g., streaming datasets with multiple splits). The new “load a single split, split in‑memory” approach is simpler, more predictable, and still satisfies all current use cases (WikiText, OpenWebMath, and any future corpus that does not ship with a curated validation set).

## Risk Mitigation

- **Regression risk**: The changes are focused on the data‑loading layer; the tokenization, grouping, and training loops remain untouched. The existing WikiText integration tests will catch any breakage.
- **Performance impact**: Streaming + conversion adds a small overhead for the first epoch, but this is negligible compared to training time. For very large `max_samples` the conversion may consume memory; we assume `max_samples` is chosen to fit in RAM.
- **Complexity**: The refactored code is significantly simpler than the previous version (removed about 50 lines of split‑handling logic). This reduces maintenance burden and makes it easier to add new text corpora.

## Next Steps

1. Apply the pending change to `run_experiments.py` (Outstanding Task 1).
2. Run the test suite to verify nothing is broken.
3. If all tests pass, run a quick end‑to‑end validation with both datasets.
4. Update the documentation.

After these steps the refactoring will be complete and ready for production experiments.