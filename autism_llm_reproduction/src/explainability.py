import torch
from transformers import AutoTokenizer
from model import AutismClassifier
import numpy as np

MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "autism_model.pth"

def predict(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.softmax(logits, dim=1)
    return probs[0][1].item() # Probability of Autism

def explain_text(text, model, tokenizer):
    # Split into sentences (simple split for demo)
    sentences = text.split(". ")
    sentences = [s.strip() for s in sentences if s.strip()]
    
    baseline_prob = predict(model, tokenizer, text)
    print(f"\nOriginal Text Probability of Autism: {baseline_prob:.4f}")
    print("-" * 50)
    
    scores = []
    for i, sent in enumerate(sentences):
        # Create a version of text without this sentence
        text_without = ". ".join([s for j, s in enumerate(sentences) if j != i])
        if not text_without: text_without = "" # Handle single sentence case
        
        prob_without = predict(model, tokenizer, text_without)
        
        # Importance = Drop in probability when removed
        # (Higher drop means sentence was important for the positive class)
        importance = baseline_prob - prob_without
        scores.append((sent, importance))
        
    # Sort by importance
    scores.sort(key=lambda x: x[1], reverse=True)
    
    print("Most Salient Sentences (Deconstructing Intuition):")
    for sent, score in scores:
        print(f"[{score:+.4f}] {sent}")

def main():
    # Load model
    model = AutismClassifier(MODEL_NAME)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
    except:
        print("Warning: Trained model not found, using initialized weights for demo.")
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Test cases
    test_text = "Patient shows repetitive hand flapping and strong interest in train schedules. Minimal eye contact observed during the interview."
    explain_text(test_text, model, tokenizer)
    
    test_text_2 = "Child communicates well with peers. No repetitive behaviors noted."
    explain_text(test_text_2, model, tokenizer)

if __name__ == "__main__":
    main()
