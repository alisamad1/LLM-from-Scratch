# Simple Transformer Language Model from Scratch

This project demonstrates how to build a minimal transformer-based language model (LLM) using PyTorch. The notebook walks through each step of the architecture, from tokenization to embeddings, positional encoding, self-attention, transformer blocks, training, and inference.

## Features

- **Step-by-step construction** of a transformer-based language model
- Educational explanations for each component
- Simple training loop on toy data
- Example inference to predict the next word

## How It Works

1. **Tokenization**  
   Converts words in a sentence to unique integer IDs.

2. **Embeddings**  
   Maps token IDs to dense vectors, allowing the model to learn word relationships.

3. **Positional Encoding**  
   Adds information about word order to the embeddings.

4. **Self-Attention**  
   Allows the model to focus on relevant words in a sequence.

5. **Transformer Block**  
   Combines self-attention and feed-forward neural networks with normalization.

6. **Language Model Assembly**  
   Stacks multiple transformer blocks and projects outputs to the vocabulary space.

7. **Training Loop**  
   Teaches the model to predict the next word in a sequence using toy data.

8. **Inference Example**  
   Predicts the next word given an input sequence.

## Usage

### Requirements

- Python 3.7+
- PyTorch

Install dependencies:
```bash
pip install torch
```

### Running the Notebook

1. Download or clone this repository.
2. Open `test.ipynb` in Jupyter Notebook, JupyterLab, or VSCode.
3. Run all cells in order to train and test the model.

### Example Output

```
Epoch 0, Loss: 1.13
Epoch 10, Loss: 0.56
...
Input: hello world, Predicted: how
```

## Project Structure

- **Step 1:** Tokenization
- **Step 2:** Embeddings Layer
- **Step 3:** Positional Encoding
- **Step 4:** Self-Attention
- **Step 5:** Transformer Block
- **Step 6:** Full Language Model Assembly
- **Step 7:** Training the Model
- **Step 8:** Using the Model for Inference

Each step is explained with markdown cells in the notebook.

## Acknowledgments

- Inspired by ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- Built with [PyTorch](https://pytorch.org/)

**Educational Note:**  
This project is for learning and demonstration purposes. It uses a very small dataset and a minimal model for clarity and simplicity.

Feel free to copy, modify, and use this README for your project! If you need a more detailed or customized README, just ask.

[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/56029749/981a2624-0143-4fca-a47b-94ce2321f24e/test.ipynb