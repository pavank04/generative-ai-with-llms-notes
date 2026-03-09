# Parameter Efficient Fine-Tuning

1. [Parameter Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
2. [PEFT techniques 1: LoRA](#peft-techniques-1-lora)
3. [PEFT techniques 2: Soft prompts](#peft-techniques-2-soft-prompts)
4. [Lab 2 walkthrough](#lab-2-walkthrough)
5. [Lab 2 - Fine-tune a generative AI model for dialogue summarization](#lab-2---fine-tune-a-generative-ai-model-for-dialogue-summarization)
6. [Week 2 quiz](#week-2-quiz)
7. [Week 2 Resources](#week-2-resources)

## Parameter Efficient Fine-Tuning (PEFT)

- Memory required for full fine-tuning apart from for model storage
  - Trainable weights
  - Optimizer states
  - Gradients
  - Forward activations
  - Temp memory
- These can take up 12-20 times weights memory.
- Types of PEFT
  - Some techniques freeze most of the model weights and focus on fine tuning a subset of existing model parameters, for example, particular layers or components.
  - Other techniques don't touch the original model weights at all, and instead add a small number of new parameters or layers and fine-tune only the new components.
- Full fine-tuning creates full copy of original LLM per task.
  - Each of these is same size as the original model.
  - Can create an expensive storage problem if fine-tuned for multiple tasks.
- PEFT fine-tuning saves space and is flexible.
- PEFT trade-offs
  - Parameter efficiency
  - Memory efficiency
  - Model performance
  - Inference costs
  - Training speed

- Three main classes of PEFT methods
  - Selective
    - Fine-tunes only a subset of the original LLM parameters.
    - Option to train only certain components of the model or specific layers, or even individual parameter types.
    - Cons
      - Performance of these methods has been found to be mixed.
      - Significant trade-offs between parameter efficiency and compute efficiency.
  - Reparameterization
    - Creates low rank transformation of the original network weights.
    - LoRA: A commonly used technique of this type
  - Additive
    - Two main approaches
      - Adapters
        - Add new trainable layers to the architecture of the model
          - Typically inside the encoder or decoder components after the attention or feed-forward layers.
      - Soft prompts
        - Keep the model architecture fixed and frozen
        - Focus on manipulating the input to achieve better performance.
          - Adding trainable parameters to the prompt enbeddings, or
          - Keeping the input fixed and retraining the embedding weights.

## PEFT techniques 1: LoRA

- Applying LoRA to self-attention layers of the model
  - Researchers have found this being often enough to fine-tune for a task and achieve performance gains.
  - Since most of the parameters of LLMs are in the attention layers, we get the biggest savings in trainable parameters by applying LoRA to these weight matrices.
- In principle, one can also use LoRA on other components like the feed-forward layers.

- Steps
  - Freeze most of the original LLM weights
  - Inject 2 **rank decomposition matrices**
  - Train the weights of the smaller matrices

- Steps to update model for inference
  - Matrix multiply the low rank matrices
  - Add to original weights

### LoRA: Low Rank Adaptation of LLMs

- Train different rank decomposition matrices for different tasks.
- Update weights before inference

## PEFT techniques 2: Soft prompts

- Prompt tuning is not prompt engineering!
- Prompt tuning adds trainable "soft prompt" to inputs
  - Soft prompt
    - The set of additional trainable tokens that can be added to prompt.
    - Supervised learning process determines their optimal values.
    - Typically 20-100 virtual tokens
- Parameters updated
  - Full fine-tuning: Millions to billions
  - Prompt tuning: 10k-100k

### Performance of prompt tuning

- As model size increases, performance of prompt tuning improves to the level of full fine tuning.

### Interpretability of soft prompts

- The trained tokens don't correspond to any known token, word or phrase in the vocabulary of the LLM.
- However, an analysis of the nearest neighbor tokens to the soft prompt location shows that they form tight semantic clusters.

## Lab 2 walkthrough

## Lab 2 - Fine-tune a generative AI model for dialogue summarization

- [Notebook](../code/Lab_2_fine_tune_generative_ai_model.ipynb)

## Week 2 quiz

## Week 2 Resources

### Multi-task, instruction fine-tuning

- [Scaling Instruction-Finetuned Language Models](https://arxiv.org/pdf/2210.11416.pdf)
- [Introducing FLAN: More generalizable Language Models with Instruction Fine-Tuning](https://ai.googleblog.com/2021/10/introducing-flan-more-generalizable.html)

### Model Evaluation Metrics

- [HELM - Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/latest/)
- [General Language Understanding Evaluation (GLUE) benchmark](https://openreview.net/pdf?id=rJ4km2R5t7)
- [SuperGLUE](https://super.gluebenchmark.com/)
- [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)
- [Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/pdf/2009.03300.pdf)
- [BigBench-Hard - Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models](https://arxiv.org/pdf/2206.04615.pdf)

### Parameter- efficient fine tuning (PEFT)

- [Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.15647.pdf)
- [On the Effectiveness of Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2211.15583.pdf)

### LoRA

- [LoRA Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf)

### Prompt tuning with soft prompts

- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/pdf/2104.08691.pdf)
