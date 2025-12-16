# TnT - The Normalized Transformer: Eigen-Learning Rates Meet Stepwise Normalization
A PyTorch implementation of a normalized transformer architecture for language modeling with enhanced stability through normalization techniques.

## 🚀 Overview
This project implements a variant of the transformer architecture that uses normalization techniques to improve training stability and performance. This implementation focuses on normalization strategies that enhance model convergence and performance.

- Designed and architected a fully normalized encoder-decoder architecture constraining all embeddings and attention matrices to unit hypersphere, eliminating weight decay and reducing parameters by 2.5%.
- Incorporated adaptive eigen learning rates across components, enabling dynamic balancing of self-attention, cross-attention, and feed-forward contributions.
- Achieved 11.31% average improvement on GLUE tasks with notable 11.5% gain on WiC and 3.79% on BoolQ, plus 7% lower validation loss under 2000 iterations

---

## ✨ Key Features

- Encoder-decoder transformer architecture with self-attention and cross-attention mechanisms
- RMSNorm for layer normalization providing better training stability than traditional LayerNorm
- Dual normalization strategies :
  - Standard normalization ```( use_nGPT=0 )```: Uses RMSNorm before each sub-layer
  - nGPT normalization ```( use_nGPT=1 )```: Uses specialized normalization with learnable scaling parameters
- Learning rate adaptation for different components (attention and MLP)
- Visualization tools for model analysis and training dynamics
- Rotary position embeddings for better handling of sequence information
---

## 🏗️ Architecture
- **Encoder**
  - Self-attention with normalization
  - Feed-forward networks with GELU activation
  - Configurable depth
- **Decoder**
  - Self-attention + cross-attention
  - Feed-forward with normalization
  - Supports **causal masking**
- **Normalization**
  - RMSNorm implementation
  - Learnable scaling in **nGPT mode**

---

## 📊 Analysis Tools
- **Learning Rate Analysis** → visualize layer-wise learning rates  
- **Embedding Norm Analysis** → monitor input/output embedding distributions  
- **Training Dynamics** → real-time model behavior visualization  

---

## Training
The model is designed for pre-training on large text corpora. Training scripts are provided in the pre-training directory
```python ntransformer/pre-training/train_ntransformer_lm_gelu.py```


---

## Inference
The model supports inference for text generation
```output = inference(model, tokenizer, input_text, max_length=128)```

---
## Requirements
- PyTorch
- NumPy
- Matplotlib
- Wandb (optional, for logging)

---
## ⚡ Quickstart
Clone the repo and install dependencies:
```bash
git clone https://github.com/tanisthahota/normalized-transformer.git
cd normalized-transformer
pip install -r requirements.txt
