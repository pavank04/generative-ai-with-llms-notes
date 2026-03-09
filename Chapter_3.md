# Fine-tuning LLMs with instruction

1. [Introduction - Week 2](#introduction---week-2)
2. [Instruction fine-tuning](#instruction-fine-tuning)
3. [Fine-tuning on a single task](#fine-tuning-on-a-single-task)
4. [Multi-task instruction fine-tuning](#multi-task-instruction-fine-tuning)
5. [Scaling instruct models](#scaling-instruct-models)
6. [Model evaluation](#model-evaluation)
7. [Benchmarks](#benchmarks)

## Introduction - Week 2

## Instruction fine-tuning

- This lesson covers
  - Methods that you can use to improve the performance of an existing model for your specific case.
  - Important metrics that can be used to evaluate the performance of your finetuned LLM and quantify its improvement over the base model you started with.

### Fine-tuning an LLM with instruction prompts

### Limitations of in-context learning

- In-context learning may not work for smaller models.
- Examples take up space in the context window.

### LLM fine-tuning at a high level

### Using prompts to fine-tune LLMs with instruction

- Each prompt/completion pair includes a specific "instruction" to the LLM
- Full fine-tuning updates all parameters

### Sample prompt instruction templates

### LLM fine-tuning process

- Use standard crossentropy function to calculate loss between the two token districutions
  - To compare the distribution of the completion and that of the training label.
- Instruct LLM
  - Fine-tuned model

## Fine-tuning on a single task

- Interestingly, good results can be achieved with relatively few examples
  - Often 500-1000 examplescan result in good performance in contrast to the billions of pieces of texts that the model saw during pretraining.

### Catastrophic forgetting

### How to avoid catastrophic forgetting

- First note that you might not have to!
- Fine-tune on **multiple tasks** at the same time
  - May require 50-100,000 examples across many tasks
    - ?? Or is it 50k - 100k
- Consider **Parameter Efficient Fine-tuning (PEFT)**

## Multi-task instruction fine-tuning

### Instruction fine-tuning with FLAN

- FLAN: Fine-tuned LAnguage Net
- T5 => FLAN-T5
- PALM => FLAN-PALM

### FLAN-T5: Fine-tuned version of pre-trained T5 model

- FLAN-T5 is a great, general purpose, instruct model
- [Scaling Instruction-Finetuned Language Models by Chung et al. 2022](https://arxiv.org/abs/2210.11416)
- Collection of tasks
  - T0-SF
  - Muffin
  - CoT (reasoning)
  - Natural instructions

### Sample FLAN-T5 prompt templates

- The template is actually comprised of several different instructions that all basically ask the model to do the same thing.

### Improving FLAN-T5's summarization capabilities

- Further fine-tune FLAN-T5 with a domain-specific instruction dataset (**[dialogsum](https://huggingface.co/datasets/knkarthick/dialogsum)**)

## Scaling instruct models

- [Scaling Instruction-Finetuned Language Models by Chung et al, 2022](https://arxiv.org/abs/2210.11416)
  - Introduces FLAN (Fine-tuned LAnguage Net), an instruction finetuning method
  - Held-out tasks
    - Some tasks were held-out during training, and they were later used to evaluate the model's performance on unseen tasks.

## Model evaluation

### Model evaluation metrics

- Unlike traditional machine learning, with large language models where is output is non-deterministic and language-based evaluation is much more challenging.
- Two widely used evaluation metrics for different tasks
  - ROUGE
    - Recall-Oriented Understudy for Gisting Evaluation
    - Used for text summarization
  - BLEU
    - Bilingual Evaluation Undersrtudy
    - Used for text translation

### LLM Evaluation - Metrics - ROUGE-1

- Unigrams are compared

### LLM Evaluation - Metrics - ROUGE-2

- Bigrams are compared

### LLM Evaluation - Metrics - ROUGE-L

- Longest common subsequence compared
- We need to take the ROUGE score values in context.
  - Only use the scores to compare the capabilities of models if the score were determined for the same task.

### LLM Evaluation - Metrics - ROUGE clipping

### LLM Evaluation - Metrics - BLEU

- BLEU metric = Avg(precision across range of n-gram sizes)

## Benchmarks

- GLUE
  - [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461)
- SuperGLUE
  - [SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems, 2019](https://arxiv.org/abs/1905.00537)

### Benchmarks for massive models

- [Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
- BIG-bench
  - [Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them, 2022](https://arxiv.org/abs/2210.09261)

### Holistic Evaluation of Language Models (HELM)
