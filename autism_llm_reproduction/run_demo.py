import torch
import sys
import os

# Add the project root to sys.path so we can import from custom_models
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "NLP-ASD-1.0.0"))
sys.path.append(project_root)

from custom_models.sentence_attention_base_pool import SentenceAttentionBERT
from transformers import AutoTokenizer

def run_demo():
    print("🚀 Initializing Demo for Autism LLM Reproduction...")
    
    # 1. Configuration
    # Using a small model for demo purposes to avoid OOM and long download times
    # In the paper they might use RoBERTa or similar, but structure is generic.
    BASE_MODEL_NAME = "prajjwal1/bert-tiny" # Very small BERT for fast testing
    REPORT_MAX_LENGTH = 4 # Batch processing size for sentences
    
    print(f"📦 Loading Model: {BASE_MODEL_NAME}...")
    
    try:
        # Initialize the custom model
        # Note: dims need to match the base model. bert-tiny is 128 dim.
        model = SentenceAttentionBERT(
            base_model_name=BASE_MODEL_NAME,
            sentence_embed_dim=128,  # bert-tiny hidden size
            sentence_weight_dim=128,
            word_embed_dim=126, # Must be divisible by 6 (unused in forward)
            word_weight_dim=128,
            report_max_length=REPORT_MAX_LENGTH
        )
        print("✅ Model initialized successfully.")
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return

    # 2. Mock Data Creation
    # Structure: (Batch Size, Num Sentences, Num Tokens)
    # Let's say Batch=1, Sentences=8, Tokens=16
    BATCH_SIZE = 1
    NUM_SENTENCES = 8
    NUM_TOKENS = 16
    
    print("🛠 Generating mock input data...")
    input_ids = torch.randint(0, 1000, (BATCH_SIZE, NUM_SENTENCES, NUM_TOKENS))
    attn_mask = torch.ones((BATCH_SIZE, NUM_SENTENCES, NUM_TOKENS))
    
    # Sentence attention mask (Batch, 1, Sentences) - optional usually but good to have
    # The MultiheadAttention expects (N, L, S) for mask or specific shapes.
    # In the code: `attn_mask=sentence_attn_mask` passed to self.attn
    # Let's just leave it None for now as it's optional in the forward signature
    
    # 3. Forward Pass
    print("🔄 Running forward pass...")
    try:
        logits, sentence_attn_weights, attn_output = model(input_ids, attn_mask=attn_mask)
        
        print("\n🎉 Forward pass complete!")
        print(f"Logits Shape: {logits.shape}") # Should be (Batch_Size,) or scalar if squeezed
        print(f"Sentence Attention Weights Shape: {sentence_attn_weights.shape}")
        print(f"Prediction Score (Logits): {logits.item():.4f}")
        
        # Check interpretability output
        print("\n🔍 Interpretability Check:")
        print("Sentence Attention Weights (First Sample):")
        print(sentence_attn_weights[0])
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()
