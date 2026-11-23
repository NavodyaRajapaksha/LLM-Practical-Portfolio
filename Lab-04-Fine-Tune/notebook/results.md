<img width="1737" height="662" alt="image" src="https://github.com/user-attachments/assets/d15bcdce-30e8-4f00-8499-79e3ffc0194e" />

## 5. Configure LoRA and generate baseline answers output

*Load the Qwen model weights using the model name selected in Question 3. Use standard LoRA on the Tesla T4. Generate answers to all eight evaluation questions before training so they can be compared with the fine-tuned answers.*
<img width="1507" height="245" alt="image" src="https://github.com/user-attachments/assets/d2e6bcb7-af55-4290-ac61-915b502d9039" />

## 6. Fine-tune with supervised learning output

*Use TRL's SFTTrainer to train the LoRA adapter on the 32 computer-shop Q&A pairs. The base weights stay frozen, and the training loss uses the assistant's answers. Save the trained adapter after training.*
<img width="1165" height="652" alt="image" src="https://github.com/user-attachments/assets/f1cf41ad-7087-408e-85df-d380b81fe5fb" />
