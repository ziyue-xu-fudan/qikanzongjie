# Reproduction: LLMs Deconstruct Clinical Intuition in Autism Diagnosis

This project aims to reproduce the methodology described in the paper:
**"Large language models deconstruct the clinical intuition behind diagnosing autism"** (Stanley et al., Cell, 2025).

**Note:** The official source code was not publicly available at the time of creation. This is a conceptual reproduction based on the paper's abstract and methodology descriptions.

## Project Goal
To use Large Language Models (LLMs) to analyze clinical reports and identify the most "salient" sentences that drive the diagnosis of autism, thereby deconstructing clinical intuition.

## Methodology (Inferred)
1.  **Data**: ~4,000 free-form health records (Simulated in this repo).
2.  **Model**: Fine-tuned LLM (e.g., BERT/RoBERTa/Llama) for binary classification (Confirmed Autism vs. Suspected/Ruled Out).
3.  **Explainability**: An architecture to score the importance of individual sentences (Salience Map / Attention).

## Structure
- `data/`: Contains dummy clinical reports for testing.
- `src/`: Source code.
  - `model.py`: Model definition.
  - `train.py`: Training script.
  - `explainability.py`: Sentence-level importance scoring.
- `requirements.txt`: Dependencies.

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python src/train.py`
3. Run explanation: `python src/explainability.py`
