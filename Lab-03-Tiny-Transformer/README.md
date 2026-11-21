# Lab Sheet 3 — Developing a Tiny Transformer Language Model

## Objective

Train a small character-level Transformer from scratch on TinyStories and evaluate its ability to predict the next character and continue a story prompt.

## Tasks and Implementation

| Task | Implementation |
| --- | --- |
| 1. Load the dataset | Loaded the first 1,000 stories from Hugging Face's `roneneldan/TinyStories` dataset, split them into 900 training and 100 validation stories, and previewed the training text. |
| 2. Preprocess the text | Built a character vocabulary and converted text to integer IDs. |
| 3. Prepare training data | Sampled batches of 32 sequences, each 64 characters long. Targets were shifted forward by one character. |
| 4. Build the Transformer | Used 64-dimensional character and position embeddings, two Transformer layers, two attention heads per layer, and a feed-forward dimension of 128. A causal mask blocked attention to future characters. |
| 5. Initialize the model | Configured cross-entropy loss and the AdamW optimizer with a learning rate of 0.001. |
| 6. Train the model | Trained for 15 epochs with 100 parameter updates per epoch using backpropagation. |
| 7. Monitor training | Recorded training and validation loss and plotted both learning curves. |
| 8. Evaluate the model | Estimated cross-entropy on 20 randomly sampled validation batches and calculated character-level perplexity. |
| 9. Generate stories | Sampled up to 200 new characters from the prompt `Once upon a time`, using the last 64 characters as context. |
| 10. Compare model settings | Compared validation loss and story continuations from the same training run after 5 and 15 epochs. Both samples used the same prompt, temperature (1.0), and random seed (42). |
| 11. Save the model | Saved the final weights, character vocabulary, and model configuration to `tiny_transformer.pt`. |
| Optional. Build a chat window | Used Gradio to accept a prompt and display a generated continuation. |

### Run the notebook

Open [LabSheet03](notebook/) in Google Colab and run the cells in order. The notebook installs `datasets` and uses CUDA when available. The optional Gradio cell installs `gradio`. Run the final cell to save and download the model checkpoint.

## Results and Evidence

| Training checkpoint | Training loss | Validation loss |
| --- | ---: | ---: |
| Epoch 5 | 1.518 | 1.531 |
| Epoch 15 | 1.420 | 1.436 |

These values come from the per-epoch training log. The notebook also contains the separate evaluation cell for reporting validation perplexity.

- **After 5 epochs:** The continuation included “Lily sayy” and “He you little day.”
- **After 15 epochs:** The continuation included “Tim says” and invented words such as “bloon” and “cowday.”
- **Supporting evidence:** The [results](notebook/) contains the Tiny Transformer Chatbot output.

## Observations

Validation loss fell by 0.095 between epochs 5 and 15, showing better next-character prediction on the sampled validation batches. Both story samples were still hard to follow. The 15-epoch sample included a recognizable phrase such as “Tim says,” but errors in spelling and sentence structure remained.

The causal mask enforced left-to-right prediction. Multi-head attention let the model use earlier characters in the context, and residual connections and layer normalization supported training.

## Learning Outcomes

- Prepared character-level inputs and targets for next-character prediction.
- Built and trained a causal Transformer in PyTorch.
- Used loss curves, perplexity, and text samples to evaluate a language model.
- Compared training checkpoints and saved the model for later use.

