import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from cr_batcher import CRBatcher

def test_prefix_hashing():
    batcher = CRBatcher()
    s1 = "Schedule the best deals on Flipkart"
    s2 = "Schedule purchase history on Google"
    s3 = "Manage customer reviews on Zara"
    assert batcher.extract_prefix_bucket(s1) == batcher.extract_prefix_bucket(s2)
    assert batcher.extract_prefix_bucket(s1) != batcher.extract_prefix_bucket(s3)

def test_conformal_quantile_length():
    batcher = CRBatcher(tau=0.90)
    src_tokens = 25
    de_safe = batcher.predict_conformal_target_length(src_tokens, "de-DE")
    zh_safe = batcher.predict_conformal_target_length(src_tokens, "zh-CN")
    assert de_safe > zh_safe
    assert de_safe >= src_tokens

def test_zero_dropped_segments():
    batcher = CRBatcher(token_budget=512, max_batch_items=52)
    sample_units = [(i, f"Action item number {i} to be processed.") for i in range(120)]
    batches = batcher.schedule(sample_units, target_lang="es-ES")
    
    received_indices = []
    for b in batches:
        for idx, text in b:
            received_indices.append(idx)
            
    assert len(received_indices) == len(sample_units)
    assert set(received_indices) == set(range(len(sample_units)))

if __name__ == "__main__":
    test_prefix_hashing()
    test_conformal_quantile_length()
    test_zero_dropped_segments()
    print("All unit tests passed successfully!")
