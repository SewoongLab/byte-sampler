import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from byte_sampling.radix_cache import RadixCacheManager

MODEL_ID = "hf-internal-testing/tiny-random-GPT2Model"

@pytest.fixture
def model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def check_structure(r1, r2, msg_prefix=""):
    """
    Recursively compares the structure and content of two results.
    Results can be tuples (trunk_logprobs, branch_logprobs) or nested dictionaries.
    """
    # Handle tuple (trunk_logprobs, branch_logprobs)
    if isinstance(r1, tuple) and isinstance(r2, tuple):
        assert len(r1) == len(r2)
        # Compare trunks (lists of floats/tensors)
        for k, (t1, t2) in enumerate(zip(r1[0], r2[0])):
            torch.testing.assert_close(t1, t2, atol=1e-5, rtol=1e-5, msg=f"{msg_prefix} Trunk mismatch at {k}")
        # Compare branches
        check_structure(r1[1], r2[1], msg_prefix)
        return

    # Handle dictionary branches
    if isinstance(r1, dict) and isinstance(r2, dict):
        assert r1.keys() == r2.keys(), f"{msg_prefix} Keys mismatch: {r1.keys()} vs {r2.keys()}"
        for k in r1:
            if k is None:
                # Leaf logprob
                torch.testing.assert_close(r1[k], r2[k], atol=1e-5, rtol=1e-5, msg=f"{msg_prefix} Value mismatch at {k}")
            else:
                check_structure(r1[k], r2[k], f"{msg_prefix} -> {k}")
        return
        
    # Fallback
    assert False, f"Type mismatch or unexpected type: {type(r1)} vs {type(r2)}"

def test_batching_equivalence(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    
    # Create dummy requests
    input_ids_1 = tokenizer.encode("Hello", return_tensors="pt")[0].tolist()
    input_ids_2 = tokenizer.encode("Hello world", return_tensors="pt")[0].tolist()
    input_ids_3 = tokenizer.encode("Test", return_tensors="pt")[0].tolist()

    # Structure: (trunk, branches)
    # branches = {last_token: {None: None}}
    def make_req(ids):
        if not ids: return ([], {})
        trunk = ids[:-1]
        last = ids[-1]
        branches = {last: {None: None}}
        return (trunk, branches)

    reqs = [make_req(input_ids_1), make_req(input_ids_2), make_req(input_ids_3)]
    
    # Run sequential (fresh manager for each to ensure independence)
    seq_results = []
    for req in reqs:
        m_seq = RadixCacheManager(model, tokenizer)
        # Note: query expects a list of requests
        res = m_seq.query([req])[0]
        seq_results.append(res)
    
    # Run batched
    m_batch = RadixCacheManager(model, tokenizer)
    batch_results = m_batch.query(reqs)
    
    assert len(batch_results) == 3
    
    for i, (seq_res, batch_res) in enumerate(zip(seq_results, batch_results)):
        check_structure(seq_res, batch_res, f"Request {i}")

def test_gc_equivalence(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    
    # Sequence of overlapping requests to build up cache
    # Using simple letters for clarity of structure
    ids = tokenizer.encode("A B C D E F G H", return_tensors="pt")[0].tolist()
    
    # Ensure we have enough tokens
    assert len(ids) >= 8
    
    # Define queries that share prefixes
    # q1: trunk=[0], branch={1} -> access 0, 1
    q1 = (ids[:1], {ids[1]: {None: None}}) 
    # q2: trunk=[0,1,2], branch={3} -> access 0, 1, 2, 3
    q2 = (ids[:3], {ids[3]: {None: None}})
    # q3: trunk=[0,1], branch={4} -> access 0, 1, 4. (2,3 are now unused)
    q3 = (ids[:2], {ids[4]: {None: None}})
    
    queries_to_run = [q1, q2, q3]
    
    # Run without GC
    m_no_gc = RadixCacheManager(model, tokenizer)
    res_no_gc = []
    for q in queries_to_run:
        res_no_gc.append(m_no_gc.query([q])[0])
        
    # Run with GC
    m_gc = RadixCacheManager(model, tokenizer)
    res_gc = []
    for q in queries_to_run:
        # Run GC after every step
        res_gc.append(m_gc.query([q], do_gc=True)[0])
    
    # Verify results identity
    for i in range(len(res_no_gc)):
        r1 = res_no_gc[i]
        r2 = res_gc[i]
        check_structure(r1, r2, f"Query {i}")

    # Verify GC effect
    # m_no_gc should have accumulated more tokens than m_gc
    # m_no_gc: 0,1,2,3,4 (plus potentially more if structure differs, but at least 5 unique content tokens)
    # m_gc: 0,1,4 (3 unique content tokens)
    
    # Access internal cache structure
    # cache_meta is list[SequenceCache]. Batch size 1.
    seq_no_gc = m_no_gc.cache_meta[0].seq
    seq_gc = m_gc.cache_meta[0].seq
    
    # Filter for non-pad tokens? 
    # The implementation pads to max length in batch. Batch size 1, so no padding usually unless forced?
    # _make_pad_token is used when batching.
    # But even if there's padding, the length of `seq` represents the allocated cache slots.
    # In m_no_gc, we kept appending. In m_gc, we rebuilt.
    
    assert len(seq_no_gc) > len(seq_gc), f"GC did not reduce cache size: {len(seq_no_gc)} vs {len(seq_gc)}"

def test_prefix_caching_single_equivalence(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    # Define sequences A and B where A is a prefix of B
    text_a = "Hello"
    text_b = "Hello world"
    
    ids_a = tokenizer.encode(text_a, return_tensors="pt")[0].tolist()
    ids_b = tokenizer.encode(text_b, return_tensors="pt")[0].tolist()

    # Helper to create request structure
    def make_req(ids):
        if not ids: return ([], {})
        trunk = ids[:-1]
        last = ids[-1]
        branches = {last: {None: None}}
        return (trunk, branches)

    req_a = make_req(ids_a)
    req_b = make_req(ids_b)

    # --- Scenario: Single Request Caching ---
    # Case 1.1: Query B directly
    m_direct_b = RadixCacheManager(model, tokenizer)
    res_direct_b = m_direct_b.query([req_b])[0]

    # Case 1.2: Query A, then Query B (should reuse cache from A)
    m_a_then_b = RadixCacheManager(model, tokenizer)
    _ = m_a_then_b.query([req_a]) # Process A, cache its prefix
    res_a_then_b = m_a_then_b.query([req_b])[0] # Process B, should use cached prefix from A

    # Compare results for B
    check_structure(res_direct_b, res_a_then_b, "Single Query B vs A then B")

def test_prefix_caching_batched_equivalence(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    # Define sequences A and B where A is a prefix of B
    text_a = "Hello"
    text_b = "Hello world"
    
    ids_a = tokenizer.encode(text_a, return_tensors="pt")[0].tolist()
    ids_b = tokenizer.encode(text_b, return_tensors="pt")[0].tolist()

    # Helper to create request structure
    def make_req(ids):
        if not ids: return ([], {})
        trunk = ids[:-1]
        last = ids[-1]
        branches = {last: {None: None}}
        return (trunk, branches)

    req_a = make_req(ids_a)
    req_b = make_req(ids_b)

    # Create a dummy request to fill another slot in the batch
    dummy_ids = tokenizer.encode("Dummy sequence", return_tensors="pt")[0].tolist()
    dummy_req = make_req(dummy_ids)

    # --- Scenario: Batched Request Caching (on a specific slot in batch) ---
    # Case 2.1: Batched Query B directly (B at index 1)
    batch_direct_b_input = [dummy_req, req_b]
    m_batch_direct_b = RadixCacheManager(model, tokenizer)
    results_batch_direct_b = m_batch_direct_b.query(batch_direct_b_input)
    res_batch_direct_b = results_batch_direct_b[1] # Get result for B (index 1)

    # Case 2.2: Batched Query A, then Batched Query B (B at index 1, should reuse cache from A)
    batch_a_input = [dummy_req, req_a]
    batch_b_input = [dummy_req, req_b]
    
    m_batch_a_then_b = RadixCacheManager(model, tokenizer)
    _ = m_batch_a_then_b.query(batch_a_input) # Process batch with A, cache its prefix in slot 1
    results_batch_a_then_b = m_batch_a_then_b.query(batch_b_input) # Process batch with B, should use cached prefix from A in slot 1
    res_batch_a_then_b = results_batch_a_then_b[1] # Get result for B (index 1)

    # Compare results for B
    check_structure(res_batch_direct_b, res_batch_a_then_b, "Batched Query B vs A then B")

def test_partial_match_caching_single_equivalence(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    # Define sequences A and B that share a prefix but diverge
    text_a = "The quick brown fox"
    text_b = "The quick brown dog"
    
    ids_a = tokenizer.encode(text_a, return_tensors="pt")[0].tolist()
    ids_b = tokenizer.encode(text_b, return_tensors="pt")[0].tolist()
    
    # Verify we actually have a shared prefix of significant length
    common_len = 0
    for x, y in zip(ids_a, ids_b):
        if x == y:
            common_len += 1
        else:
            break
    assert common_len >= 2, f"Sequences do not share enough common prefix: {ids_a} vs {ids_b}"
    assert ids_a != ids_b, "Sequences must be different"

    # Helper to create request structure
    def make_req(ids):
        if not ids: return ([], {})
        trunk = ids[:-1]
        last = ids[-1]
        branches = {last: {None: None}}
        return (trunk, branches)

    req_a = make_req(ids_a)
    req_b = make_req(ids_b)

    # --- Scenario: Single Request Caching (Partial Match) ---
    # Case 1.1: Query B directly
    m_direct_b = RadixCacheManager(model, tokenizer)
    res_direct_b = m_direct_b.query([req_b])[0]

    # Case 1.2: Query A, then Query B (should reuse shared prefix)
    m_a_then_b = RadixCacheManager(model, tokenizer)
    _ = m_a_then_b.query([req_a]) # Process A, cache prefix
    res_a_then_b = m_a_then_b.query([req_b])[0] # Process B, should reuse shared prefix

    # Compare results for B
    check_structure(res_direct_b, res_a_then_b, "Single Query B vs A then B (Partial Match)")

def test_partial_match_caching_batched_equivalence(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    # Define sequences A and B that share a prefix but diverge
    text_a = "The quick brown fox"
    text_b = "The quick brown dog"
    
    ids_a = tokenizer.encode(text_a, return_tensors="pt")[0].tolist()
    ids_b = tokenizer.encode(text_b, return_tensors="pt")[0].tolist()

    # Helper to create request structure
    def make_req(ids):
        if not ids: return ([], {})
        trunk = ids[:-1]
        last = ids[-1]
        branches = {last: {None: None}}
        return (trunk, branches)

    req_a = make_req(ids_a)
    req_b = make_req(ids_b)

    # Create a dummy request to fill another slot in the batch
    dummy_ids = tokenizer.encode("Dummy sequence", return_tensors="pt")[0].tolist()
    dummy_req = make_req(dummy_ids)

    # --- Scenario: Batched Request Caching (Partial Match) ---
    # Case 2.1: Batched Query B directly
    batch_direct_b_input = [dummy_req, req_b]
    m_batch_direct_b = RadixCacheManager(model, tokenizer)
    results_batch_direct_b = m_batch_direct_b.query(batch_direct_b_input)
    res_batch_direct_b = results_batch_direct_b[1] 

    # Case 2.2: Batched Query A, then Batched Query B (Reuse shared prefix)
    batch_a_input = [dummy_req, req_a]
    batch_b_input = [dummy_req, req_b]
    
    m_batch_a_then_b = RadixCacheManager(model, tokenizer)
    _ = m_batch_a_then_b.query(batch_a_input) 
    results_batch_a_then_b = m_batch_a_then_b.query(batch_b_input) 
    res_batch_a_then_b = results_batch_a_then_b[1] 

    # Compare results for B
    check_structure(res_batch_direct_b, res_batch_a_then_b, "Batched Query B vs A then B (Partial Match)")

def test_complex_tree_structure(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    
    # Construct a tree:
    # Trunk: "The"
    #        |-> " quick" -> " brown"
    #        |-> " lazy"
    
    trunk_str = "The"
    branch1_str = " quick"
    branch1_child_str = " brown"
    branch2_str = " lazy"
    
    trunk_ids = tokenizer.encode(trunk_str, return_tensors="pt")[0].tolist()
    b1_ids = tokenizer.encode(branch1_str, add_special_tokens=False)
    b1_child_ids = tokenizer.encode(branch1_child_str, add_special_tokens=False)
    b2_ids = tokenizer.encode(branch2_str, add_special_tokens=False)
    
    # Build tree dict
    # Structure: {token_id: {child_token_id: ...}}
    # We want logprobs at the end of each path
    
    # Path 1: " The" -> " quick" -> " brown"
    # The trunk covers "The".
    # Branches starts with first token of " quick".
    
    # Constructing the nested dictionary manually
    # deeply nested: b1 -> b1_child -> None
    curr = {None: None}
    for tid in reversed(b1_child_ids):
        curr = {tid: curr}
    branch1_tree = curr
    
    # merge b1_ids into branch1_tree
    # branch1_tree currently represents the structure AFTER b1_ids. 
    # We need to prepend b1_ids.
    curr = branch1_tree
    for tid in reversed(b1_ids):
        curr = {tid: curr}
    full_branch1 = curr
    
    # Path 2: " The" -> " lazy"
    curr = {None: None}
    for tid in reversed(b2_ids):
        curr = {tid: curr}
    full_branch2 = curr
    
    # Merge branches at the root level (which corresponds to end of Trunk)
    # Note: simple merge assuming first tokens are different.
    combined_branches = {}
    combined_branches.update(full_branch1)
    combined_branches.update(full_branch2)
    
    assert len(combined_branches) == 2, "First tokens of branches must be distinct for this simple merge"

    req_tree = (trunk_ids, combined_branches)
    
    # Execute Tree Query
    mgr = RadixCacheManager(model, tokenizer)
    res_tree = mgr.query([req_tree])[0]
    
    # Execute Linear Queries for verification
    # Path 1: Verify against a single query with the same trunk but only Path 1's branch
    # Note: We must use the same trunk so that the branch logprob accumulation covers the same tokens.
    req_1 = (trunk_ids, full_branch1)
    mgr_1 = RadixCacheManager(model, tokenizer)
    res_1 = mgr_1.query([req_1])[0]
    
    # Path 2: Verify against a single query with the same trunk but only Path 2's branch
    req_2 = (trunk_ids, full_branch2)
    mgr_2 = RadixCacheManager(model, tokenizer)
    res_2 = mgr_2.query([req_2])[0]
    
    # Verify Path 1 values
    # Navigate res_tree to get value for Path 1
    # res_tree is (trunk_logprobs, branches_logprobs)
    _, branches_tree = res_tree
    
    val_tree_1 = branches_tree
    for tid in b1_ids: val_tree_1 = val_tree_1[tid]
    for tid in b1_child_ids: val_tree_1 = val_tree_1[tid]
    val_tree_1 = val_tree_1[None]
    
    _, branches_1 = res_1
    val_linear_1 = branches_1
    for tid in b1_ids: val_linear_1 = val_linear_1[tid]
    for tid in b1_child_ids: val_linear_1 = val_linear_1[tid]
    val_linear_1 = val_linear_1[None]
    
    torch.testing.assert_close(val_tree_1, val_linear_1, atol=1e-5, rtol=1e-5, msg="Tree Path 1 mismatch")
    
    # Verify Path 2 values
    val_tree_2 = branches_tree
    for tid in b2_ids: val_tree_2 = val_tree_2[tid]
    val_tree_2 = val_tree_2[None]
    
    _, branches_2 = res_2
    val_linear_2 = branches_2
    for tid in b2_ids: val_linear_2 = val_linear_2[tid]
    val_linear_2 = val_linear_2[None]
    
    torch.testing.assert_close(val_tree_2, val_linear_2, atol=1e-5, rtol=1e-5, msg="Tree Path 2 mismatch")


def test_batch_independent_evolution(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    
    # Define two distinct sequences
    text_a = "Sequence A is evolving quickly"
    text_b = "Sequence B moves slowly but surely"
    
    ids_a = tokenizer.encode(text_a, return_tensors="pt")[0].tolist()
    ids_b = tokenizer.encode(text_b, return_tensors="pt")[0].tolist()
    
    # Define split points (prefix vs full)
    split_a = len(ids_a) // 2
    split_b = len(ids_b) // 2
    
    ids_prefix_a = ids_a[:split_a]
    ids_prefix_b = ids_b[:split_b]
    
    def make_req(ids):
        if not ids: return ([], {})
        trunk = ids[:-1]
        last = ids[-1]
        branches = {last: {None: None}}
        return (trunk, branches)
    
    req_prefix_a = make_req(ids_prefix_a)
    req_full_a = make_req(ids_a)
    
    req_prefix_b = make_req(ids_prefix_b)
    req_full_b = make_req(ids_b)
    
    # Reference: Query full sequences directly
    m_ref = RadixCacheManager(model, tokenizer)
    res_ref = m_ref.query([req_full_a, req_full_b])
    
    # Test: Evolve independently
    m_batch = RadixCacheManager(model, tokenizer)
    
    # Step 1: Initialize prefixes
    _ = m_batch.query([req_prefix_a, req_prefix_b])
    
    # Step 2: Extend A, maintain B (re-query prefix)
    # This checks if A can extend using its cached prefix while B stays put (refreshing its prefix cache)
    res_step2 = m_batch.query([req_full_a, req_prefix_b])
    
    # Verify A matches full reference
    check_structure(res_ref[0], res_step2[0], "Step 2: A extended")
    
    # Step 3: Maintain A (re-query full), Extend B
    # This checks if B can extend using its cached prefix (from step 1, refreshed in step 2)
    res_step3 = m_batch.query([req_full_a, req_full_b])
    
    # Verify B matches full reference
    check_structure(res_ref[1], res_step3[1], "Step 3: B extended")
    
    # Also verify A still matches (stability)
    check_structure(res_ref[0], res_step3[0], "Step 3: A stability")


def test_eviction_and_resurrection(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    
    # Init with warning enabled to test that path (though we catch the warning or just verify correctness)
    mgr = RadixCacheManager(model, tokenizer, warn_on_resurrection=True)
    
    ids_a = tokenizer.encode("Sequence A", return_tensors="pt")[0].tolist()
    ids_b = tokenizer.encode("Sequence B is completely different", return_tensors="pt")[0].tolist()
    
    def make_req(ids):
        if not ids: return ([], {})
        trunk = ids[:-1]
        last = ids[-1]
        branches = {last: {None: None}}
        return (trunk, branches)
    
    req_a = make_req(ids_a)
    req_b = make_req(ids_b)
    
    # 1. Query A (Populates cache)
    res_a_1 = mgr.query([req_a])[0]
    
    # 2. Query B with GC (Should evict A's unique tokens as they are not used in B)
    # Note: query([req_b], do_gc=True) will run B, then run GC.
    # B uses its own tokens. A's tokens have gc_gen=1. B's tokens will have gc_gen=2.
    # GC will keep gen 2.
    _ = mgr.query([req_b], do_gc=True)
    
    # 3. Query A again.
    # The tree structure for A still exists in mgr (in cache_meta children pointers), 
    # but the cache slots should have been reclaimed or invalidated.
    # This triggers "resurrection" logic: finding a node in the tree but not in the active sequence.
    # We expect a warning and correct results.
    with pytest.warns(UserWarning, match="Found resurrected token"):
        res_a_2 = mgr.query([req_a])[0]
        
    check_structure(res_a_1, res_a_2, "Resurrected A vs Original A")