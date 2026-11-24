# Final Reflection

The six labs covered sentiment classification, LLM-assisted product design, transformer training, fine-tuning, retrieval and AI agents. A recurring lesson was that the way a system is evaluated determines which errors are visible. Accuracy, validation loss and successful demonstrations each provide different kinds of evidence.

## From classification to text generation

In [Lab 01](Lab-01-Machine-Learning/README.md), logistic regression with a Bag of Words representation achieved 78.73% test accuracy on sentiment classification. This provided a baseline for judging how much can be learned from word frequencies. The representation discards word order, which limits its ability to interpret phrases whose meaning depends on context.

In [Lab 03](Lab-03-Tiny-Transformer/README.md), building a character-level transformer connected causal attention to next-character prediction. Validation loss fell from 1.531 at epoch 5 to 1.436 at epoch 15, while both sampled story continuations remained difficult to follow. Reading the generated text exposed spelling and sentence structure problems that the loss values alone could not describe.

## Designing an application with an LLM

In [Lab 02](Lab-02-LLM-Everyday-Life/README.md), the Pocket Path group activity turned the problem of tracking small student expenses into a budgeting app concept. The Quick Add mock-up reduced the entry to an amount and a category. This exercise showed how an LLM can support the move from a broad problem to a concrete workflow. A prototype test with students could show whether the shorter form makes it easier to record everyday purchases consistently.

## Fine-tuning and retrieving evidence

In [Lab 04](Lab-04-Fine-Tune/README.md), LoRA increased key-fact matches from 1 out of 8 to 3 out of 8 in the fictional computer shop evaluation. The fine-tuned model gave the correct 14-day return window but omitted conditions and invented a worldwide shipping policy. A stronger evaluation would check each required condition and include questions the model should answer by acknowledging missing information.

In [Lab 05](Lab-05-RAG-Pipeline/README.md), the university assistant displayed retrieved PDF names and page numbers, making its answers easier to verify. Some details in the activity summary still needed checking against the original document. The multilingual demo also showed that Sinhala and Tamil questions could receive English answers. A fair comparison of retrieval strategies would use the same questions and PDFs and check whether each strategy retrieved the expected passage before assessing answer accuracy and language.

## Building agents that act

In [Lab 06](Lab-06-AI-Agents/README.md), the agent combined tool selection, memory, university document retrieval and Google Calendar access. A working calendar demo showed how the agent could act on a request. Correct operation also depends on choosing the appropriate tool and passing the correct arguments. A practical safeguard would be to show the proposed event title, date and time for confirmation before creating it.

## Next steps

The module developed my practical skills in preparing data, training models, comparing outputs, retrieving evidence and integrating tools. The next step would be to build a reusable evaluation set for the university assistant with expected answers, source pages and questions the documents cannot answer. This would make it possible to measure whether a change improves retrieval, improves answer accuracy or helps the assistant recognize when evidence is missing.

