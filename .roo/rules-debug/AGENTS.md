# Project Debug Rules (Non-Obvious Only)

- **Debugging CUDA Issues**: When CUDA is not available, tests will fail with `assert torch.cuda.is_available()`. Check GPU drivers and CUDA installation
- **Memory Leaks**: Integration tests require explicit `gc.collect()` calls due to large model memory requirements
- **Test Failures**: Some edge case tests are marked `@pytest.mark.xfail` due to known limitations
- **Logging**: Detailed debugging output is available when GPU is available, showing device placement and memory usage
- **Test Data**: Edge case tests include extreme tensor shapes and numerical values that may cause instability
- **Performance**: Tests include performance benchmarks that may fail on slower hardware