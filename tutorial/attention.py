import torch
import math
import pytest
import time


# ============= Part 1: Base Classes (Already Implemented) =============

class BaseAttention:
    """Base class for attention implementations"""
    def __init__(self):
        pass

    def attention(self, q, k, v, scale=None):
        """
        Standard attention: softmax(scale * Q @ K^T) @ V
        scale defaults to 1/sqrt(d)
        """
        d = q.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(d)
        s = self.softmax((q @ k.T) * scale, dim=1)
        return s @ v

    def softmax(self, input, dim):
        raise NotImplementedError("softmax not implemented")


class NativeAttention(BaseAttention):
    """Standard softmax attention (reference)"""
    def softmax(self, input, dim):
        return torch.softmax(input, dim)


class SafeAttention(BaseAttention):
    """Numerically stable softmax attention with max trick"""
    def softmax(self, input, dim):
        '''
        softmax with numerical stability: exp(x - max(x))
        
        Args:
            input: attention scores matrix, shape (N, N) or (rows, cols)
            dim: dimension along which to apply softmax
        
        HINT: 
        1. Find row-wise max of input using torch.max(input, dim=dim)
           -> Shape should be (rows, 1) if keepdim=True
        2. Subtract max from input: exp(x - max(x)) for stability
           -> Shape remains (rows, cols) due to broadcasting
        3. Compute exp of the shifted input
           -> Shape: (rows, cols)
        4. Sum exp values row-wise to get denominator
           -> Shape: (rows, 1)
        5. Return normalized result: numerator / denominator
           -> Shape: (rows, cols)
        
        Key insight: exp(x - max(x)) prevents overflow since result <= exp(0) = 1
        '''
        raise NotImplementedError("Implement SafeAttention.softmax")


# ============= Part 2: Flash Attention (Main Task) =============

class OnlineSafeAttention(BaseAttention):
    '''
    Flash Attention Implementation using online algorithm
    
    Key Idea: Process attention in blocks, updating running statistics
    (max value m, denominator d, output o) as we process each K block.
    
    This avoids materializing the full attention matrix in memory.
    '''
    
    def __init__(self, BLOCK_M=4):
        super().__init__()
        self.BLOCK_M = BLOCK_M

    def attention(self, q, k, v, device='cpu', scale=None):
        """
        Main entry point - calls v2_multihead which is the most general
        """
        return self.attention_v2_multihead(q, k, v, device, scale)

    def attention_v1(self, q, k, v, device='cpu', scale=None):
        '''
        Flash Attention v1: Iterate over K blocks in outer loop
        
        Shape: q, k, v all (seq_len, head_dim), denoted as (N, d)
        
        HINT - Algorithm:
        1. Initialize output buffers:
           - o: output accumulator, shape (N, d)
           - m: running max, shape (N, 1)
           - l: running denominator, shape (N, 1)
           All initialized to 0, -inf, 0 respectively
        
        2. Split q, k, v into blocks of size BLOCK_M
           -> Resulting blocks will have shape (BLOCK_M, d)
        
        3. Outer loop over K blocks (j):
           For each K_block and V_block (shape (BLOCK_M, d)):
               Inner loop over Q blocks (i):
                   For each Q_block (shape (BLOCK_M, d)):
                       - Compute S = (Q_block @ K_block^T) * scale
                         -> Shape: (BLOCK_M, BLOCK_M)
                       - Compute m_local = max(S, dim=-1, keepdim=True)
                         -> Shape: (BLOCK_M, 1)
                       - Update m_new = max(m_old, m_local)
                         -> Shape: (BLOCK_M, 1) (m_old comes from slicing global m)
                       - Compute P = exp(S - m_new)  [numerically stable]
                         -> Shape: (BLOCK_M, BLOCK_M)
                       - Compute l_local = sum(P, dim=-1, keepdim=True)
                         -> Shape: (BLOCK_M, 1)
                       - Update l_new = l_old * exp(m_old - m_new) + l_local * exp(m_local - m_new)
                         -> Shape: (BLOCK_M, 1)
                       - Update o_new = (o_old * exp(m_old - m_new) + P @ V_block)
                         -> Shape: (BLOCK_M, d)
                       
                       Note: We keep o unnormalized and normalize at the end
                       
                       - Store back m_new, l_new, o_new for next iteration
                         -> Update global tensors at indices [i*BLOCK:(i+1)*BLOCK]
        
        4. Return o / l (normalize by denominator)
           -> Shape: (N, d)
        
        Key insight: By maintaining m and l, we can safely accumulate outputs
        from different K blocks even though they use different max values.
        '''
        d = q.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(d)
        
        raise NotImplementedError("Implement attention_v1")

    def attention_v2(self, q, k, v, device='cpu', scale=None):
        '''
        Flash Attention v2: Iterate over Q blocks in outer loop (more efficient)
        
        Shape: q, k, v all (seq_len, head_dim), denoted as (N, d)
        
        HINT - Algorithm:
        1. Split q, k, v into blocks of size BLOCK_M
        
        2. Create output list to store results
        
        3. Outer loop over Q blocks (j):
           For each Q_block (shape (BLOCK_M, d)):
               - Initialize (for THIS Q_block only):
                 * m_local = -inf, shape (BLOCK_M, 1)
                 * l_local = 0, shape (BLOCK_M, 1)
                 * o_local = 0, shape (BLOCK_M, d)
               
               Inner loop over K blocks (i):
                   For each K_block and V_block (shape (BLOCK_M, d)):
                       - Compute S = (Q_block @ K_block^T) * scale
                         -> Shape: (BLOCK_M, BLOCK_M)
                       - Compute m_new = max(m_local, max(S, dim=-1, keepdim=True))
                         -> Shape: (BLOCK_M, 1)
                       - Compute P = exp(S - m_new)
                         -> Shape: (BLOCK_M, BLOCK_M)
                       - Compute l_local_new = sum(P, dim=-1, keepdim=True)
                         -> Shape: (BLOCK_M, 1)
                       - Update l_new = l_local * exp(m_local - m_new) + l_local_new
                       - Update o_new = o_local * exp(m_local - m_new) + P @ V_block
                         -> Shape: (BLOCK_M, d)
                       - Update m_local, l_local, o_local for next K block
               
               - Normalize this Q_block's output: o_local / l_local
               - Append to results
        
        4. Concatenate all Q block results
           -> Final shape: (N, d)
        
        Advantage over v1: Each Q block accumulates completely before moving on,
        which is more cache-friendly and requires less memory for intermediate storage.
        '''
        d = q.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(d)
        
        raise NotImplementedError("Implement attention_v2")

    def attention_v2_multihead(self, q, k, v, device='cpu', scale=None):
        '''
        Flash Attention v2 with multi-head support
        
        Shape: q, k, v are (batch_size, num_heads, seq_len, head_dim)
               denoted as (B, H, N, d)
        
        HINT - Key considerations:
        1. Similar to v2 but handles batch and head dimensions
        2. The algorithm remains the same, but applied independently for each (batch, head) pair
        3. Use torch.split() to split along seq_len dimension (dim=-2)
           -> Blocks will be (B, H, BLOCK_M, d)
        4. For indexing output: use output_buffer[..., j*BLOCK_M:(j+1)*BLOCK_M, :]
           The "..." automatically handles all leading dimensions (batch, head)
        5. For matrix multiply: remember to transpose K correctly
           Q @ K^T means q @ k.transpose(-2, -1)
           -> Q: (..., BLOCK, d), K.T: (..., d, BLOCK) -> Result: (..., BLOCK, BLOCK)
        
        Broadcasting: torch operations automatically broadcast over batch and head dims,
        so the same code works for both single-head and multi-head cases!
        '''
        d = q.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(d)
        
        raise NotImplementedError("Implement attention_v2_multihead")


# ============= Part 3: Utility Functions =============

def get_tensors(BS, HEAD, SEQLEN, DIM, dtype=torch.float32, device='cpu'):
    """Generate random Q, K, V tensors"""
    q = torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    k = torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    v = torch.empty((BS, HEAD, SEQLEN, DIM), dtype=dtype, device=device).normal_(mean=0.0, std=0.5)
    return q, k, v


def extract_head(x, batch_idx=0, head_idx=0):
    """
    Safely extract a single head from multi-dimensional tensor
    x: (..., seq_len, head_dim) 
    Returns: (seq_len, head_dim)
    """
    return x[batch_idx, head_idx]


# ============= Part 4: Test Suite =============

class TestFlashAttention:
    """Comprehensive test suite for Flash Attention implementations"""
    
    # ===== Fixtures =====
    
    @pytest.fixture
    def tiny_inputs(self):
        """Tiny inputs for debugging: (1, 1, 8, 4)"""
        return get_tensors(1, 1, 8, 4, dtype=torch.float32, device='cpu')
    
    @pytest.fixture
    def small_inputs(self):
        """Small inputs: (2, 2, 16, 8)"""
        return get_tensors(2, 2, 16, 8, dtype=torch.float32, device='cpu')
    
    @pytest.fixture
    def medium_inputs(self):
        """Medium inputs: (4, 8, 64, 64)"""
        return get_tensors(4, 8, 64, 64, dtype=torch.float32, device='cpu')
    
    @pytest.fixture
    def single_head_tiny(self):
        """Single head tiny: (1, 1, 8, 4)"""
        return get_tensors(1, 1, 8, 4, dtype=torch.float32, device='cpu')
    
    # ===== Reference Implementation =====
    
    def reference_attention_4d(self, q, k, v, scale=None):
        """
        Standard PyTorch attention with proper scaling: softmax(Q @ K^T / sqrt(d)) @ V
        
        Args:
            q, k, v: (batch, heads, seq_len, head_dim)
            scale: optional scaling factor (defaults to 1/sqrt(d))
        Returns:
            output: (batch, heads, seq_len, head_dim)
        """
        d = q.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(d)
        
        # Compute attention scores with proper scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        # Apply softmax
        attn_weights = torch.softmax(scores, dim=-1)
        # Apply to values
        output = torch.matmul(attn_weights, v)
        return output
    
    def reference_attention_2d(self, q, k, v, scale=None):
        """
        Standard PyTorch attention for 2D input: (seq_len, head_dim)
        """
        d = q.shape[-1]
        if scale is None:
            scale = 1.0 / math.sqrt(d)
        
        scores = (q @ k.T) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        output = attn_weights @ v
        return output
    
    # ===== Test 1: SafeAttention Correctness (10 points) =====
    
    def test_01_safe_attention_vs_native(self, single_head_tiny):
        """
        Test 1: SafeAttention produces same result as NativeAttention [10 points]
        
        Tests: Numerical stability with max trick
        """
        q, k, v = single_head_tiny
        # Extract single head: (seq_len, head_dim)
        q_2d = extract_head(q, 0, 0)
        k_2d = extract_head(k, 0, 0)
        v_2d = extract_head(v, 0, 0)
        
        native = NativeAttention()
        safe = SafeAttention()
        
        native_result = native.attention(q_2d, k_2d, v_2d)
        safe_result = safe.attention(q_2d, k_2d, v_2d)
        
        assert torch.allclose(native_result, safe_result, atol=1e-5), \
            f"Max diff: {(native_result - safe_result).abs().max():.2e}"
    
    # ===== Test 2: v1 Shape (5 points) =====
    
    def test_02_v1_output_shape(self, small_inputs):
        """Test 2: v1 produces correct output shape [5 points]"""
        q, k, v = small_inputs
        # Extract first head
        q_2d = extract_head(q, 0, 0)
        k_2d = extract_head(k, 0, 0)
        v_2d = extract_head(v, 0, 0)
        
        attn = OnlineSafeAttention(BLOCK_M=4)
        output = attn.attention_v1(q_2d, k_2d, v_2d, device='cpu')
        
        assert output.shape == v_2d.shape, f"Expected {v_2d.shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
    
    # ===== Test 3: v1 Numerical Stability (5 points) =====
    
    def test_03_v1_numerical_stability(self, small_inputs):
        """Test 3: v1 maintains numerical stability [5 points]"""
        q, k, v = small_inputs
        q_2d = extract_head(q, 0, 0)
        k_2d = extract_head(k, 0, 0)
        v_2d = extract_head(v, 0, 0)
        
        attn = OnlineSafeAttention(BLOCK_M=4)
        output = attn.attention_v1(q_2d, k_2d, v_2d, device='cpu')
        
        assert torch.all(torch.isfinite(output)), "Output contains non-finite values (NaN or Inf)"
    
    # ===== Test 4: v1 vs Safe Attention (15 points) =====
    
    def test_04_v1_correctness_vs_safe(self, small_inputs):
        """Test 4: v1 matches SafeAttention output [15 points]"""
        q, k, v = small_inputs
        q_2d = extract_head(q, 0, 0)
        k_2d = extract_head(k, 0, 0)
        v_2d = extract_head(v, 0, 0)
        
        safe = SafeAttention()
        attn = OnlineSafeAttention(BLOCK_M=4)
        
        safe_result = safe.attention(q_2d, k_2d, v_2d)
        v1_result = attn.attention_v1(q_2d, k_2d, v_2d, device='cpu')
        
        max_diff = (safe_result - v1_result).abs().max().item()
        
        score = 0
        if max_diff <= 0.01:
            score = 15
        elif max_diff <= 0.05:
            score = 10
        elif max_diff <= 0.1:
            score = 5
        
        print(f"\nTest 4 - v1 vs SafeAttention: Max diff = {max_diff:.2e}, Score: {score}/15")
        assert torch.allclose(safe_result, v1_result, atol=1e-2), \
            f"Max diff: {max_diff:.2e}"
    
    # ===== Test 5: v2 Shape (5 points) =====
    
    def test_05_v2_output_shape(self, small_inputs):
        """Test 5: v2 produces correct output shape [5 points]"""
        q, k, v = small_inputs
        q_2d = extract_head(q, 0, 0)
        k_2d = extract_head(k, 0, 0)
        v_2d = extract_head(v, 0, 0)
        
        attn = OnlineSafeAttention(BLOCK_M=4)
        output = attn.attention_v2(q_2d, k_2d, v_2d, device='cpu')
        
        assert output.shape == v_2d.shape
    
    # ===== Test 6: v2 vs v1 (10 points) =====
    
    def test_06_v2_matches_v1(self, small_inputs):
        """Test 6: v2 produces same result as v1 [10 points]"""
        q, k, v = small_inputs
        q_2d = extract_head(q, 0, 0)
        k_2d = extract_head(k, 0, 0)
        v_2d = extract_head(v, 0, 0)
        
        attn = OnlineSafeAttention(BLOCK_M=4)
        v1_result = attn.attention_v1(q_2d, k_2d, v_2d, device='cpu')
        v2_result = attn.attention_v2(q_2d, k_2d, v_2d, device='cpu')
        
        max_diff = (v1_result - v2_result).abs().max().item()
        print(f"\nTest 6 - v2 vs v1: Max diff = {max_diff:.2e}")
        
        assert torch.allclose(v1_result, v2_result, atol=1e-3), \
            f"Max diff: {max_diff:.2e}"
    
    # ===== Test 7: v2 Correctness (15 points) =====
    
    def test_07_v2_correctness_vs_reference(self, small_inputs):
        """Test 7: v2 matches reference implementation [15 points]"""
        q, k, v = small_inputs
        q_2d = extract_head(q, 0, 0)
        k_2d = extract_head(k, 0, 0)
        v_2d = extract_head(v, 0, 0)
        
        ref = self.reference_attention_2d(q_2d, k_2d, v_2d)
        
        attn = OnlineSafeAttention(BLOCK_M=4)
        v2_result = attn.attention_v2(q_2d, k_2d, v_2d, device='cpu')
        
        max_diff = (ref - v2_result).abs().max().item()
        
        score = 0
        if max_diff <= 0.01:
            score = 15
        elif max_diff <= 0.05:
            score = 10
        elif max_diff <= 0.1:
            score = 5
        
        print(f"\nTest 7 - v2 vs Reference: Max diff = {max_diff:.2e}, Score: {score}/15")
        assert torch.allclose(ref, v2_result, atol=1e-2)
    
    # ===== Test 8: Multihead Shape (5 points) =====
    
    def test_08_multihead_output_shape(self, small_inputs):
        """Test 8: multihead produces correct output shape [5 points]"""
        q, k, v = small_inputs
        
        attn = OnlineSafeAttention(BLOCK_M=4)
        output = attn.attention_v2_multihead(q, k, v, device='cpu')
        
        assert output.shape == v.shape, f"Expected {v.shape}, got {output.shape}"
    
    # ===== Test 9: Multihead vs 2D (10 points) =====
    
    def test_09_multihead_batch_head_processing(self, small_inputs):
        """Test 9: multihead correctly processes all batches and heads [10 points]"""
        q, k, v = small_inputs
        batch, heads, seq, dim = q.shape
        
        attn = OnlineSafeAttention(BLOCK_M=4)
        output = attn.attention_v2_multihead(q, k, v, device='cpu')
        
        # Check individual heads match 2D processing
        max_diff_overall = 0.0
        for b in range(batch):
            for h in range(heads):
                q_2d = extract_head(q, b, h)
                k_2d = extract_head(k, b, h)
                v_2d = extract_head(v, b, h)
                
                expected_2d = attn.attention_v2(q_2d, k_2d, v_2d, device='cpu')
                actual_2d = output[b, h]
                
                max_diff = (expected_2d - actual_2d).abs().max().item()
                max_diff_overall = max(max_diff_overall, max_diff)
                
                assert torch.allclose(expected_2d, actual_2d, atol=1e-3), \
                    f"Mismatch at batch {b}, head {h}, max_diff={max_diff:.2e}"
        
        print(f"\nTest 9 - Multihead vs 2D: Max diff = {max_diff_overall:.2e}")
    
    # ===== Test 10: Multihead Correctness (15 points) =====
    
    def test_10_multihead_correctness(self, small_inputs):
        """Test 10: multihead matches reference implementation [15 points]"""
        q, k, v = small_inputs
        
        ref = self.reference_attention_4d(q, k, v)
        
        attn = OnlineSafeAttention(BLOCK_M=4)
        output = attn.attention_v2_multihead(q, k, v, device='cpu')
        
        max_diff = (ref - output).abs().max().item()
        
        score = 0
        if max_diff <= 0.01:
            score = 15
        elif max_diff <= 0.05:
            score = 10
        elif max_diff <= 0.1:
            score = 5
        
        print(f"\nTest 10 - Multihead vs Reference: Max diff = {max_diff:.2e}, Score: {score}/15")
        assert torch.allclose(ref, output, atol=1e-2)
    
    # ===== Test 11: Block Size Consistency (5 points) =====
    
    def test_11_different_block_sizes_consistent(self, medium_inputs):
        """Test 11: Different BLOCK_M sizes give same result [5 points]"""
        q, k, v = medium_inputs
        
        attn_b4 = OnlineSafeAttention(BLOCK_M=4)
        attn_b8 = OnlineSafeAttention(BLOCK_M=8)
        
        output_b4 = attn_b4.attention_v2_multihead(q, k, v, device='cpu')
        output_b8 = attn_b8.attention_v2_multihead(q, k, v, device='cpu')
        
        max_diff = (output_b4 - output_b8).abs().max().item()
        print(f"\nTest 11 - Block consistency (BLOCK_M=4 vs 8): Max diff = {max_diff:.2e}")
        
        assert torch.allclose(output_b4, output_b8, atol=1e-3), \
            f"Max diff: {max_diff:.2e}"
    
    # ===== Test 12: Main Integration Test (5 points) =====
    
    def test_12_main_attention_method(self, small_inputs):
        """Test 12: Main attention() method works correctly [5 points]"""
        q, k, v = small_inputs
        
        attn = OnlineSafeAttention(BLOCK_M=4)
        output = attn.attention(q, k, v, device='cpu')
        
        assert output.shape == v.shape
        assert torch.all(torch.isfinite(output))


# ===== Part 5: Performance Comparison =====

def benchmark_attention(name, attn_fn, q, k, v, num_runs=5):
    """
    Benchmark an attention function
    Returns: avg_time_ms, output
    """
    # Warmup
    with torch.no_grad():
        for _ in range(2):
            _ = attn_fn(q, k, v)
    
    # Timing
    times = []
    output = None
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            output = attn_fn(q, k, v)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = sum(times) / len(times)
    print(f"{name:25s}: {avg_time:8.4f} ms")
    return avg_time, output


def run_performance_analysis():
    """Compare performance of all implementations"""
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS: Flash Attention Improvements")
    print("="*70)
    
    # Test configurations
    configs = [
        ("Small", 1, 1, 32, 16),
        ("Medium", 2, 4, 128, 64),
        ("Large", 4, 8, 512, 64),
    ]
    
    for config_name, BS, HEAD, SEQLEN, DIM in configs:
        print(f"\nConfig: {config_name} (BS={BS}, Head={HEAD}, Seq={SEQLEN}, Dim={DIM})")
        print("-" * 70)
        
        q, k, v = get_tensors(BS, HEAD, SEQLEN, DIM, dtype=torch.float32, device='cpu')
        
        # Native attention
        native_attn = NativeAttention()
        
        def native_fn(q, k, v):
            # Extract single head for fair comparison
            q_2d = extract_head(q, 0, 0)
            k_2d = extract_head(k, 0, 0)
            v_2d = extract_head(v, 0, 0)
            return native_attn.attention(q_2d, k_2d, v_2d)
        
        # Safe attention
        safe_attn = SafeAttention()
        
        def safe_fn(q, k, v):
            q_2d = extract_head(q, 0, 0)
            k_2d = extract_head(k, 0, 0)
            v_2d = extract_head(v, 0, 0)
            return safe_attn.attention(q_2d, k_2d, v_2d)
        
        # Flash attention versions
        flash = OnlineSafeAttention(BLOCK_M=max(4, SEQLEN//4))
        
        def flash_v1_fn(q, k, v):
            q_2d = extract_head(q, 0, 0)
            k_2d = extract_head(k, 0, 0)
            v_2d = extract_head(v, 0, 0)
            return flash.attention_v1(q_2d, k_2d, v_2d, device='cpu')
        
        def flash_v2_fn(q, k, v):
            q_2d = extract_head(q, 0, 0)
            k_2d = extract_head(k, 0, 0)
            v_2d = extract_head(v, 0, 0)
            return flash.attention_v2(q_2d, k_2d, v_2d, device='cpu')
        
        def flash_multihead_fn(q, k, v):
            return flash.attention_v2_multihead(q, k, v, device='cpu')
        
        # Run benchmarks
        time_native, out_native = benchmark_attention("Native Attention", native_fn, q, k, v)
        time_safe, out_safe = benchmark_attention("Safe Attention", safe_fn, q, k, v)
        
        try:
            time_v1, out_v1 = benchmark_attention("Flash Attn v1", flash_v1_fn, q, k, v)
            speedup_v1 = time_native / time_v1
            print(f"{'Speedup (v1 vs Native)':25s}: {speedup_v1:8.2f}x")
        except NotImplementedError:
            print(f"{'Flash Attn v1':25s}: NOT IMPLEMENTED")
        
        try:
            time_v2, out_v2 = benchmark_attention("Flash Attn v2", flash_v2_fn, q, k, v)
            speedup_v2 = time_native / time_v2
            speedup_v2_vs_v1 = time_v1 / time_v2 if 'time_v1' in locals() else 0
            print(f"{'Speedup (v2 vs Native)':25s}: {speedup_v2:8.2f}x")
            if speedup_v2_vs_v1 > 0:
                print(f"{'Speedup (v2 vs v1)':25s}: {speedup_v2_vs_v1:8.2f}x")
        except NotImplementedError:
            print(f"{'Flash Attn v2':25s}: NOT IMPLEMENTED")
        
        try:
            time_mh, out_mh = benchmark_attention("Flash Attn v2-MH", flash_multihead_fn, q, k, v)
            speedup_mh = time_native / time_mh
            print(f"{'Speedup (v2-MH vs Native)':25s}: {speedup_mh:8.2f}x")
        except NotImplementedError:
            print(f"{'Flash Attn v2-MH':25s}: NOT IMPLEMENTED")


# ===== Part 6: Demo Main =====

def main():
    """Demo/testing script"""
    print("\n" + "="*70)
    print("Flash Attention Lab - Demonstration")
    print("="*70)
    
    # Run performance analysis
    run_performance_analysis()
    
    print("\n" + "="*70)
    print("To run all tests: pytest test_flash_attention.py -v")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()