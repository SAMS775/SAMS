
# SAMS: Sign-Aligned Multi-Bias Vector Subtraction for Debiasing Large Language Models

This repository contains the code for our AAAI 2026 submission on structural bias mitigation in large language models using SAMS.

## 🔧 Setup

Install dependencies:
```bash
pip install -r requirements.txt
```
### Run Full Pipeline
```bash
python AggressiveBBQSDUTrainer.py
```
This runs the entire pipeline:
- Fine-tunes LoRA adapters on multiple BBQ bias axes
- Extracts task vectors
- Merges them via sign-consistent alignment
- Applies the final debiasing vector ∆⋆ to the base model
- Evaluates on BBQ (bias score + perplexity)

### Directory Structure
```bash
supplementary/
├── DataPreprocessing.py 
├── SAMSPipeline.py       
├── README.md                       
├── data/
│   ├── race/race_forget_set.json
│   ├── race/race_eval_data.json
│   ├── race/race_retain_set.json
│   └── (each category)
```
