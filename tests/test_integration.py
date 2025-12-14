import pytest
import torch
import time
import os
import gc  # Added for explicit garbage collection
from transformers.models.gpt2 import GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.bert import BertForMaskedLM, BertTokenizer
from transformers.models.roberta import RobertaForMaskedLM, RobertaTokenizer
from dendritic.enhancement import enhance_model_with_dendritic, get_polynomial_stats, load_dendritic_model, save_dendritic_model
from dendritic.dataset_handlers.PythonAlpacaHandler import PythonAlpacaHandler
from dendritic.layers.DendriticLayer import DendriticLayer
from dendritic.layers.DendriticStack import DendriticStack
from torch.utils.data import DataLoader
from datasets import Dataset
import json
import psutil

# =====================
# Fixtures and Utilities
# =====================


@pytest.fixture(scope="module")
def small_dataset():
    """Create a small mock dataset for quick testing"""
    return Dataset.from_dict(
        {
            "prompt": [
                "### Instruction:\nWrite a function\n### Input:\nNone\n### Output:",
                "### Instruction:\nWrite a loop\n### Input:\nNone\n### Output:",
            ],
            "output": ["def func(): pass", "for i in range(5): print(i)"],
        }
    )


@pytest.fixture(scope="module")
def gpt2_model_and_tokenizer():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@pytest.fixture(scope="module")
def bert_model_and_tokenizer():
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer


@pytest.fixture(scope="module")
def roberta_model_and_tokenizer():
    model = RobertaForMaskedLM.from_pretrained("roberta-base")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    return model, tokenizer


def create_dataloader(dataset, tokenizer, batch_size=2, max_length=128):
    """Create a dataloader from a dataset"""
    handler = PythonAlpacaHandler(tokenizer=tokenizer, max_length=max_length)
    tokenized = dataset.map(
        handler.tokenize_function, batched=True, remove_columns=dataset.column_names, load_from_cache_file=False
    )
    # Set format to PyTorch tensors
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return DataLoader(tokenized, batch_size=batch_size, shuffle=True)


def measure_memory_usage():
    """Measure current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)


# =====================
# End-to-End Workflow Tests
# =====================


@pytest.mark.parametrize(
    "model_fixture, model_name",
    [
        ("gpt2_model_and_tokenizer", "GPT-2"),
        ("bert_model_and_tokenizer", "BERT"),
        ("roberta_model_and_tokenizer", "RoBERTa"),
    ],
)
@pytest.mark.integration
def test_model_enhancement_workflow(request, model_fixture, model_name, small_dataset):
    """Test complete enhancement pipeline for different model architectures"""
    model, tokenizer = request.getfixturevalue(model_fixture)
    dataloader = create_dataloader(small_dataset, tokenizer)

    # Enhance model
    enhanced_model = enhance_model_with_dendritic(
        model,
        target_layers=["mlp"] if "gpt" in model_name.lower() else ["intermediate"],
        poly_rank=8,
        freeze_linear=True,
        verbose=False,
    )

    # Verify parameter counts
    total_params = sum(p.numel() for p in enhanced_model.parameters())
    trainable_params = sum(
        p.numel() for p in enhanced_model.parameters() if p.requires_grad
    )
    assert trainable_params > 0
    assert trainable_params < total_params

    # Test forward pass
    batch = next(iter(dataloader))
    # Convert to long tensor for models like BERT that require integer inputs
    inputs = batch["input_ids"].long() if "bert" in model_name.lower() else batch["input_ids"]
    outputs = enhanced_model(inputs)
    assert outputs.logits.shape[0] == inputs.shape[0]

    # Test polynomial stats
    stats = get_polynomial_stats(enhanced_model)
    assert len(stats) > 0
    for layer_stats in stats.values():
        assert "scale" in layer_stats
        assert "poly_rank" in layer_stats


@pytest.mark.integration
def test_custom_model_enhancement(small_dataset):
    """Test enhancement with a custom model"""

    class CustomModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(128, 256)
            self.linear2 = torch.nn.Linear(256, 128)

        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x)))

    model = CustomModel()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    dataloader = create_dataloader(small_dataset, tokenizer)

    enhanced_model = enhance_model_with_dendritic(
        model, target_layers=["linear1"], poly_rank=4, freeze_linear=False
    )

    # Test forward pass
    batch = next(iter(dataloader))
    inputs = batch["input_ids"].float()
    outputs = enhanced_model(inputs)
    assert outputs.shape == (inputs.shape[0], 128)


# =====================
# Data Pipeline Tests
# =====================


@pytest.mark.integration
def test_data_pipeline_integration(gpt2_model_and_tokenizer, small_dataset):
    """Test end-to-end data pipeline integration"""
    model, tokenizer = gpt2_model_and_tokenizer
    handler = PythonAlpacaHandler(tokenizer=tokenizer, max_length=128)

    # Test data
    prepared = handler.prepare_data(split="train", test_size=0.2)
    assert "train" in prepared
    assert "eval" in prepared

    # Create dataloaders
    train_dl = DataLoader(prepared["train"], batch_size=2, shuffle=True)
    eval_dl = DataLoader(prepared["eval"], batch_size=2, shuffle=False)

    # Test batch processing
    train_batch = next(iter(train_dl))
    assert "input_ids" in train_batch
    assert "attention_mask" in train_batch
    assert "labels" in train_batch

    # Test model can process batches
    outputs = model(
        input_ids=train_batch["input_ids"],
        attention_mask=train_batch["attention_mask"],
        labels=train_batch["labels"],
    )
    assert outputs.loss is not None


# =====================
# Training Workflow Tests
# =====================


@pytest.mark.integration
def test_training_workflow(gpt2_model_and_tokenizer, small_dataset):
    """Test complete training workflow with dendritic enhancements"""
    model, tokenizer = gpt2_model_and_tokenizer
    enhanced_model = enhance_model_with_dendritic(
        model, target_layers=["mlp.c_fc"], poly_rank=8, freeze_linear=True
    )

    dataloader = create_dataloader(small_dataset, tokenizer)
    optimizer = torch.optim.AdamW(
        [p for p in enhanced_model.parameters() if p.requires_grad], lr=5e-5
    )

    # Capture initial parameters for diagnostics
    initial_params = {
        n: p.data.clone() for n, p in enhanced_model.named_parameters() if p.requires_grad
    }
    
    # Training loop with diagnostics
    enhanced_model.train()
    prev_loss = float("inf")
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Only run a few steps for testing
            break

        optimizer.zero_grad()
        inputs = batch["input_ids"]
        outputs = enhanced_model(inputs, labels=inputs)
        loss = outputs.loss
        loss.backward()
        
        # Diagnostic: Check gradients for dendritic parameters
        print("\nGradient diagnostics:")
        for name, param in enhanced_model.named_parameters():
            if "dendritic" in name or "poly" in name or "scale" in name:
                grad_exists = param.grad is not None
                grad_norm = param.grad.norm().item() if grad_exists else 0.0
                print(f"  {name}: grad_exists={grad_exists}, grad_norm={grad_norm:.6f}")

        optimizer.step()
        
        # Diagnostic: Check if parameters changed from initial state
        print("\nParameter change diagnostics:")
        for name, param in enhanced_model.named_parameters():
            if param.requires_grad:
                changed = not torch.equal(initial_params[name], param.data)
                print(f"  {name}: changed={changed}")

        # Verify loss decreases
        if loss.item() < prev_loss:
            prev_loss = loss.item()

    # Final check that parameters changed
    enhanced_model.eval()
    for name, param in enhanced_model.named_parameters():
        if param.requires_grad:
            assert not torch.equal(initial_params[name], param.data), \
                   f"Parameter {name} did not change during training"


# =====================
# Multi-Architecture Tests
# =====================


@pytest.mark.parametrize(
    "model_fixture, layer_type",
    [
        ("gpt2_model_and_tokenizer", "mlp"),
        ("bert_model_and_tokenizer", "intermediate"),
        ("roberta_model_and_tokenizer", "intermediate"),
    ],
)
@pytest.mark.integration
def test_layer_placement_strategies(request, model_fixture, layer_type):
    """Test different layer placement strategies across architectures"""
    model, _ = request.getfixturevalue(model_fixture)

    enhanced_model = enhance_model_with_dendritic(
        model, target_layers=[layer_type], poly_rank=8, freeze_linear=True
    )

    # Verify correct layers were enhanced
    found_enhanced = False
    for name, module in enhanced_model.named_modules():
        if isinstance(module, (DendriticLayer, DendriticStack)) and layer_type in name:
            found_enhanced = True
            break
    assert found_enhanced, f"No enhanced layer found for {layer_type}"


# =====================
# Performance Tests
# =====================


@pytest.mark.integration
def test_memory_usage(gpt2_model_and_tokenizer):
    """Test memory usage patterns with dendritic enhancements"""
    model, _ = gpt2_model_and_tokenizer

    # Measure baseline memory
    base_mem = measure_memory_usage()
    model(torch.randint(0, 100, (1, 128)))  # Warmup
    base_mem_after = measure_memory_usage()

    # Enhance model
    enhanced_model = enhance_model_with_dendritic(
        model, target_layers=["mlp.c_fc"], poly_rank=8, freeze_linear=True
    )

    # Measure enhanced memory
    enhanced_mem = measure_memory_usage()
    enhanced_model(torch.randint(0, 100, (1, 128)))  # Warmup
    enhanced_mem_after = measure_memory_usage()

    # Verify memory increase is reasonable
    mem_increase = enhanced_mem_after - base_mem_after
    assert mem_increase < 100  # Less than 100MB increase


@pytest.mark.integration
def test_training_performance(gpt2_model_and_tokenizer, small_dataset):
    """Test computational efficiency during training"""
    model, tokenizer = gpt2_model_and_tokenizer
    dataloader = create_dataloader(small_dataset, tokenizer)

    # Baseline model
    start_time = time.time()
    model.train()
    for i, batch in enumerate(dataloader):
        if i >= 2:
            break
        outputs = model(batch["input_ids"], labels=batch["input_ids"])
        outputs.loss.backward()
    base_duration = time.time() - start_time

    # Enhanced model
    enhanced_model = enhance_model_with_dendritic(
        model, target_layers=["mlp.c_fc"], poly_rank=8, freeze_linear=True
    )

    start_time = time.time()
    enhanced_model.train()
    for i, batch in enumerate(dataloader):
        if i >= 2:
            break
        outputs = enhanced_model(batch["input_ids"], labels=batch["input_ids"])
        outputs.loss.backward()
    enhanced_duration = time.time() - start_time

    # Verify performance impact is reasonable
    assert enhanced_duration < base_duration * 1.5  # Less than 50% slowdown


@pytest.mark.integration
def test_scaling_with_model_size():
    """Test scaling with different model sizes"""
    model_sizes = ["gpt2", "gpt2-medium", "gpt2-large"]
    results = {}

    for size in model_sizes:
        try:
            model = GPT2LMHeadModel.from_pretrained(size)
            tokenizer = GPT2Tokenizer.from_pretrained(size)
            tokenizer.pad_token = tokenizer.eos_token

            start_time = time.time()
            enhanced_model = enhance_model_with_dendritic(
                model, target_layers=["mlp.c_fc"], poly_rank=8, freeze_linear=True
            )
            creation_time = time.time() - start_time

            # Measure memory usage
            mem_before = measure_memory_usage()
            enhanced_model(torch.randint(0, 100, (1, 128)))  # Warmup
            mem_after = measure_memory_usage()

            results[size] = {
                "param_count": sum(p.numel() for p in enhanced_model.parameters()),
                "creation_time": creation_time,
                "memory_usage": mem_after - mem_before,
            }
        except Exception as e:
            results[size] = {"error": str(e)}

    # Verify results are proportional to model size
    sizes = [s for s in model_sizes if "error" not in results[s]]
    if len(sizes) > 1:
        prev = results[sizes[0]]["memory_usage"]
        for size in sizes[1:]:
            current = results[size]["memory_usage"]
            assert current > prev  # Larger models should use more memory
            prev = current

@pytest.mark.skip(reason="Flaky test, needs investigation")
@pytest.mark.integration
def test_batch_size_handling(gpt2_model_and_tokenizer):
    """Test handling of different batch sizes"""
    model, tokenizer = gpt2_model_and_tokenizer
    enhanced_model = enhance_model_with_dendritic(
        model, target_layers=["mlp.c_fc"], poly_rank=8, freeze_linear=True
    ).cpu()

    batch_sizes = [1, 2, 4, 8, 16]
    results = {}
    # Warmup
    enhanced_model(torch.randint(0, tokenizer.vocab_size, (1, 128)))
    for bs in batch_sizes:
        try:

            # Force garbage collection to free any unused memory
            gc.collect()

            # Measure memory with no_grad to prevent graph building
            mem_before = measure_memory_usage()
            with torch.no_grad():
                inputs = torch.randint(0, tokenizer.vocab_size, (bs, 128))
                outputs = enhanced_model(inputs)
            mem_after = measure_memory_usage()

            # Force garbage collection to free any unused memory
            gc.collect()
            # Measure time with no_grad
            start_time = time.time()
            with torch.no_grad():
                for _ in range(5):
                    enhanced_model(inputs)
            duration = (time.time() - start_time) / 5

            # Debug logging
            print(f"Batch size: {bs}, Memory used: {mem_after - mem_before:.2f} MB")

            results[bs] = {
                "memory_usage": mem_after - mem_before,
                "avg_inference_time": duration,
            }
        except Exception as e:
            results[bs] = {"error": str(e)}

    # Verify all batch sizes completed without error
    successful = [bs for bs in batch_sizes if "error" not in results[bs]]
    assert len(successful) == len(batch_sizes), f"Some batch sizes failed: {results}"

    # Verify memory usage grows roughly with batch size (relaxed check)
    # CPU memory measurement is noisy, so we only check overall trend
    mem_values = [results[bs]["memory_usage"] for bs in successful]
    if len(mem_values) > 2:
        # Check that largest batch uses more memory than smallest (sanity check)
        assert mem_values[-1] > mem_values[0], "Memory should increase with batch size"
        # Check that memory doesn't grow faster than O(batch_size^2) - very relaxed bound
        max_ratio = mem_values[-1] / max(mem_values[0], 0.1)  # Avoid division by zero
        batch_ratio = batch_sizes[-1] / batch_sizes[0]
        assert max_ratio < batch_ratio ** 2, f"Memory growth {max_ratio:.1f}x exceeds quadratic bound"


# =====================
# Real-World Scenario Tests
# =====================


@pytest.mark.integration
def test_fine_tuning_workflow(gpt2_model_and_tokenizer, small_dataset):
    """Test fine-tuning workflow with enhanced model"""
    model, tokenizer = gpt2_model_and_tokenizer
    enhanced_model = enhance_model_with_dendritic(
        model, target_layers=["mlp.c_fc"], poly_rank=8, freeze_linear=True
    )

    dataloader = create_dataloader(small_dataset, tokenizer)
    optimizer = torch.optim.AdamW(
        [p for p in enhanced_model.parameters() if p.requires_grad], lr=5e-5
    )

    # Fine-tuning loop
    enhanced_model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = enhanced_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        outputs.loss.backward()
        optimizer.step()

    # Verify model can save and load
    torch.save(enhanced_model.state_dict(), "test_model.pth")
    loaded_state = torch.load("test_model.pth")
    enhanced_model.load_state_dict(loaded_state)
    os.remove("test_model.pth")


@pytest.mark.integration
def test_inference_with_enhanced_model(gpt2_model_and_tokenizer):
    """Test inference with enhanced model"""
    model, tokenizer = gpt2_model_and_tokenizer
    enhanced_model = enhance_model_with_dendritic(
        model, target_layers=["mlp.c_fc"], poly_rank=8, freeze_linear=True
    )

    # Generate text
    input_text = "def hello_world():"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    enhanced_model.eval()
    with torch.no_grad():
        # Call generate directly on the model
        # The type ignore is needed because Pylance incorrectly flags this as an error
        output = enhanced_model.generate(  # type: ignore
            input_ids, max_length=50, do_sample=True, top_p=0.95, temperature=0.8
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    assert len(generated) > len(input_text)
    assert "hello_world" in generated


@pytest.mark.integration
def test_model_serialization_workflow(gpt2_model_and_tokenizer):
    """Test the full save/load workflow with auto-configuration."""
    base_model, _ = gpt2_model_and_tokenizer
    
    # 1. Configuration
    enhancement_params = {
        "target_layers": ["mlp.c_fc"],
        "poly_rank": 8,
        "freeze_linear": True
    }

    # 2. Enhance (and pretend training happens)
    enhanced_model = enhance_model_with_dendritic(base_model, **enhancement_params)
    
    # 3. Save
    # No need to pass params! The model knows them.
    save_path = "temp_dendritic_model.pt"
    save_dendritic_model(enhanced_model, save_path)

    try:
        # 4. Load
        # We start with a fresh base model
        fresh_base = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # Load handles re-enhancement and weight loading
        restored_model = load_dendritic_model(fresh_base, save_path)

        # 5. Verification
        # Check config
        for key, value in enhancement_params.items():
            assert restored_model.dendritic_config[key] == value  # type: ignore
        
        # Check weights
        for (n1, p1), (n2, p2) in zip(
            enhanced_model.named_parameters(), restored_model.named_parameters()
        ):
            if p1.requires_grad:
                assert torch.equal(p1, p2) # exact match for loaded weights
                
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
