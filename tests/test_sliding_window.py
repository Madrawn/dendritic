"""
Unit tests for sliding window functionality in TextCorpusHandler.
"""

import pytest
from dendritic.dataset_handlers.TextCorpusHandler import sliding_window_group_texts_func


@pytest.mark.unit
def test_sliding_window_no_overlap():
    """Test sliding window with seq_stride=0 (no overlap)."""
    examples = {"input_ids": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]}

    result = sliding_window_group_texts_func(examples, max_seq_len=3, seq_stride=0)

    assert len(result["input_ids"]) == 3
    assert result["input_ids"][0] == [1, 2, 3]
    assert result["input_ids"][1] == [4, 5, 6]
    assert result["input_ids"][2] == [7, 8, 9]


@pytest.mark.unit
def test_sliding_window_max_overlap():
    """Test sliding window with seq_stride=1 (maximum overlap)."""
    examples = {"input_ids": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]}

    result = sliding_window_group_texts_func(examples, max_seq_len=3, seq_stride=1)

    assert len(result["input_ids"]) == 8
    assert result["input_ids"][0] == [1, 2, 3]
    assert result["input_ids"][1] == [2, 3, 4]
    assert result["input_ids"][2] == [3, 4, 5]
    assert result["input_ids"][3] == [4, 5, 6]
    assert result["input_ids"][4] == [5, 6, 7]
    assert result["input_ids"][5] == [6, 7, 8]
    assert result["input_ids"][6] == [7, 8, 9]
    assert result["input_ids"][7] == [8, 9, 10]


@pytest.mark.unit
def test_sliding_window_partial_overlap():
    """Test sliding window with seq_stride=2 (partial overlap)."""
    examples = {"input_ids": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]}

    result = sliding_window_group_texts_func(examples, max_seq_len=3, seq_stride=2)

    assert len(result["input_ids"]) == 4
    assert result["input_ids"][0] == [1, 2, 3]
    assert result["input_ids"][1] == [3, 4, 5]
    assert result["input_ids"][2] == [5, 6, 7]
    assert result["input_ids"][3] == [7, 8, 9]


@pytest.mark.unit
def test_sliding_window_small_sequence():
    """Test sliding window with sequence smaller than max_seq_len."""
    examples = {"input_ids": [[1, 2]]}

    result = sliding_window_group_texts_func(examples, max_seq_len=3, seq_stride=1)

    assert len(result["input_ids"]) == 0


@pytest.mark.unit
def test_sliding_window_exact_fit():
    """Test sliding window with sequence exactly fitting max_seq_len."""
    examples = {"input_ids": [[1, 2, 3]]}

    result = sliding_window_group_texts_func(examples, max_seq_len=3, seq_stride=1)

    assert len(result["input_ids"]) == 1
    assert result["input_ids"][0] == [1, 2, 3]


@pytest.mark.unit
def test_sliding_window_multiple_columns():
    """Test sliding window with multiple columns in examples."""
    examples = {"input_ids": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], "labels": [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]}

    result = sliding_window_group_texts_func(examples, max_seq_len=3, seq_stride=1)

    assert len(result["input_ids"]) == 8
    assert len(result["labels"]) == 8
    assert result["labels"][0] == [1, 2, 3]
    assert result["labels"][1] == [2, 3, 4]


@pytest.mark.unit
def test_sliding_window_empty_input():
    """Test sliding window with empty input."""
    examples = {"input_ids": []}

    result = sliding_window_group_texts_func(examples, max_seq_len=3, seq_stride=1)

    assert result == {}


@pytest.mark.unit
def test_sliding_window_multiple_sequences():
    """Test sliding window with multiple sequences in batch."""
    examples = {"input_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]}

    result = sliding_window_group_texts_func(examples, max_seq_len=3, seq_stride=1)

    # After concatenation: [1,2,3,4,5,6,7,8,9,10]
    # With stride=1: 8 chunks total
    assert len(result["input_ids"]) == 8
    assert result["input_ids"][0] == [1, 2, 3]
    assert result["input_ids"][1] == [2, 3, 4]
    assert result["input_ids"][7] == [8, 9, 10]  # Last chunk


if __name__ == "__main__":
    pytest.main([__file__])
