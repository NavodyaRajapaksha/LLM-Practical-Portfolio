# Lab Sheet 4 — Fine-Tuning an LLM with LoRA

## Objective

Fine-tune Qwen2.5-0.5B-Instruct for a fictional Sri Lankan computer shop and compare its answers before and after training.

## Tasks and Implementation

1. Install Transformers, Datasets, PEFT, TRL, Accelerate, and bitsandbytes in Google Colab. Verify CUDA availability. The run used a Tesla T4 with 14.56 GiB of GPU memory.
2. Load the Qwen tokenizer and model weights. Create 32 training Q&A pairs and eight evaluation questions.
3. Generate baseline answers, then use PEFT and TRL's `SFTTrainer` to train a LoRA adapter for eight epochs at a learning rate of `3e-4`. About 0.44% of the model parameters were trainable.
4. Ask both models the same eight questions using identical generation settings. Save the adapter to `bytebridge_lora/` and export the comparison to `lab4_comparison.csv`.

## Results and Evidence

The [LabSheet04](./notebook) contains the code, GPU check, training output, and side-by-side model answers.

| Question group | Original key-fact matches | Fine-tuned key-fact matches |
| --- | ---: | ---: |
| Seen questions | 1/2 | 2/2 |
| Rephrased questions | 0/5 | 1/5 |
| Unknown policy | 0/1 | 0/1 |
| **Total** | **1/8** | **3/8** |

## Observations

- The fine-tuned model gave the correct 14-day return window but omitted the unopened-product and receipt conditions. It also claimed the shop shipped worldwide, although the training data provided no international shipping policy.
- The key-fact check gave credit for the correct pickup duration of three days, even though the answer counted from ordering instead of the ready notification.
- In Colab, installing the missing `trl` package and removing the incompatible `torchao` version resolved the training setup errors.

## Learning Outcomes

- Configure a GPU runtime and prepare a domain-specific Q&A dataset.
- Fine-tune an instruction-tuned model with LoRA and save its adapter.
- Compare baseline and fine-tuned answers and identify errors that automated scoring can miss.
