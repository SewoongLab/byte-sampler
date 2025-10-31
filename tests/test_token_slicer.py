import pytest
import torch
from byte_sampling.byte_conditioning import ByteConditioning


@pytest.fixture
def sample_vrev():
    """Sample vocabulary reverse mapping for testing."""
    return {
        1: b"apple",
        2: b"application",
        3: b"apply",
        4: b"banana",
        5: b"band",
        6: b"banter",
        7: b"cat",
        8: b"dog",
        9: b"doggy",
        10: b"dogma",
        99: b"hello",  # Single token that doesn't match any prefix
    }


@pytest.fixture
def token_slicer(sample_vrev):
    """Create a TokenSlicer instance for testing."""
    return ByteConditioning.TokenSlicer(sample_vrev)


def test_basic_query(token_slicer):
    """Test basic query functionality without strict parameter."""
    # Query for tokens starting with "app"
    result = token_slicer.query(b"app")
    expected_tids = torch.tensor([1, 2, 3])  # apple, application, apply
    assert torch.equal(torch.sort(result).values, torch.sort(expected_tids).values)

    # Query for tokens starting with "ban"
    result = token_slicer.query(b"ban")
    expected_tids = torch.tensor([4, 5, 6])  # banana, band, banter
    assert torch.equal(torch.sort(result).values, torch.sort(expected_tids).values)

    # Query for tokens starting with "dog"
    result = token_slicer.query(b"dog")
    expected_tids = torch.tensor([8, 9, 10])  # dog, doggy, dogma
    assert torch.equal(torch.sort(result).values, torch.sort(expected_tids).values)


def test_empty_prefix(token_slicer):
    """Test querying with empty prefix returns all tokens."""
    result = token_slicer.query(b"")
    all_tids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99])
    assert torch.equal(torch.sort(result).values, torch.sort(all_tids).values)


def test_nonexistent_prefix(token_slicer):
    """Test querying with a prefix that doesn't match any tokens."""
    result = token_slicer.query(b"xyz")
    assert len(result) == 0

    result = token_slicer.query(b"elephant")
    assert len(result) == 0


def test_exact_match_included_by_default(token_slicer):
    """Test that exact matches are included by default (strict=False behavior)."""
    # Query for tokens starting with "cat" - "cat" itself should be included
    result = token_slicer.query(b"cat")
    assert 7 in result  # "cat" token should be in result
    assert len(result) == 1  # Only "cat" should match

    # Query for tokens starting with "dog" - "dog" itself should be included
    result = token_slicer.query(b"dog")
    assert 8 in result  # "dog" token should be in result
    assert len(result) == 3  # "dog", "doggy", "dogma" should be in result


def test_strict_false_explicit(token_slicer):
    """Test strict=False explicitly (should behave the same as default)."""
    # Query for tokens starting with "cat" with strict=False
    result = token_slicer.query(b"cat", strict=False)
    assert 7 in result  # "cat" token should be included
    assert len(result) == 1

    # Query for tokens starting with "dog" with strict=False
    result = token_slicer.query(b"dog", strict=False)
    assert 8 in result  # "dog" token should be included
    assert len(result) == 3  # "dog", "doggy", "dogma" should be in result


def test_strict_true_excludes_exact_match(token_slicer):
    """Test strict=True excludes exact matches."""
    # Query for tokens starting with "dog" with strict=True - "dog" should be excluded
    result = token_slicer.query(b"dog", strict=True)
    assert 8 not in result  # "dog" token should NOT be in result
    assert 9 in result  # "doggy" should be in result
    assert 10 in result  # "dogma" should be in result
    assert len(result) == 2  # Only "doggy" and "dogma" should be in result

    # Query for tokens starting with "cat" with strict=True - "cat" should be excluded
    result = token_slicer.query(b"cat", strict=True)
    assert 7 not in result  # "cat" token should NOT be in result
    assert len(result) == 0  # No tokens should remain since "cat" was the only match

    # Query for tokens starting with "app" with strict=True - "apple" should not be excluded
    # because "apple" doesn't exactly match the prefix "app"
    result = token_slicer.query(b"app", strict=True)
    assert (
        1 in result
    )  # "apple" should still be in result (doesn't exactly match "app")
    assert 2 in result  # "application" should still be in result
    assert 3 in result  # "apply" should still be in result
    assert len(result) == 3  # All three should still be included


def test_strict_true_with_real_exact_match(token_slicer):
    """Test strict=True with a prefix that exactly matches a token."""
    # Query for tokens starting with "hello" with strict=True - "hello" should be excluded
    result = token_slicer.query(b"hello", strict=True)
    assert 99 not in result  # "hello" token should NOT be in result
    assert len(result) == 0  # No tokens should remain since "hello" was the only match

    result = token_slicer.query(b"hello", strict=False)
    assert 99 in result  # "hello" token should be in result
    assert len(result) == 1  # "hello" should be the only match


def test_strict_behavior_with_prefix_subset(token_slicer):
    """Test that strict only applies to exact prefix matches, not subset matches."""
    # Query for tokens starting with "a" with strict=True
    result = token_slicer.query(b"a", strict=True)
    expected_tids = torch.tensor([1, 2, 3])  # apple, application, apply
    assert torch.equal(torch.sort(result).values, torch.sort(expected_tids).values)
    # All tokens starting with 'a' should be included since none exactly match 'a'

    # Query for tokens starting with "a" with strict=False
    result2 = token_slicer.query(b"a", strict=False)
    assert torch.equal(torch.sort(result).values, torch.sort(result2).values)
    # Results should be identical since there's no exact match for 'a'


def test_all_method(token_slicer):
    """Test the all() method returns all token IDs."""
    result = token_slicer.all()
    expected_tids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99])
    assert torch.equal(torch.sort(result).values, torch.sort(expected_tids).values)


def test_empty_prefix(token_slicer):
    """Test the all() method returns all token IDs."""
    result = token_slicer.query(b"")
    expected_tids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 99])
    assert torch.equal(torch.sort(result).values, torch.sort(expected_tids).values)

def test_edge_cases(token_slicer):
    """Test edge cases for TokenSlicer."""
    # Test with single character tokens in vocabulary
    simple_vrev = {1: b"a", 2: b"b", 3: b"aa", 4: b"ab"}
    simple_slicer = ByteConditioning.TokenSlicer(simple_vrev)

    # Query with single character that matches exact token
    result = simple_slicer.query(b"a", strict=False)
    assert 1 in result  # "a" should be included
    assert 3 in result  # "aa" should be included
    assert 4 in result  # "ab" should be included
    assert len(result) == 3

    result = simple_slicer.query(b"a", strict=True)
    assert 1 not in result  # "a" should NOT be included
    assert 3 in result  # "aa" should be included
    assert 4 in result  # "ab" should be included
    assert len(result) == 2
