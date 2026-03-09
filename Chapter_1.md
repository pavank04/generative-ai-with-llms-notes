# Introduction to LLMs and generative AI project lifecycle

1. [Course introduction](#course-introduction)
2. [Contributor Acknowledgements](#contributor-acknowledgements)
3. [Introduction - Week 1](#introduction---week-1)
4. [Generative AI & LLMs](#generative-ai--llms)
5. [LLM use cases and tasks](#llm-use-cases-and-tasks)
6. [Text generation before transformers](#text-generation-before-transformers)
7. [Transformers architecture](#transformers-architecture)
8. [Generating text with transformers](#generating-text-with-transformers)
9. [Transformers: Attention is all you need](#transformers-attention-is-all-you-need)
10. [Prompting and prompt engineering](#prompting-and-prompt-engineering)
11. [Generative configuration](#generative-configuration)
12. [Generative AI project lifecycle](#generative-ai-project-lifecycle)
13. [Introduction to AWS labs](#introduction-to-aws-labs)
14. [Lab 1 walkthrough](#lab-1-walkthrough)
15. [Lab 1 - Generative AI Use Case: Summarize Dialogue](#lab-1---generative-ai-use-case-summarize-dialogue)

## Course introduction

- This course takes a deep dive into how LLM technology actually works that includes the following technical details:
  - model training
  - instruction tuning
  - fine-tuning
  - generative AI project life cycle framework

## Contributor Acknowledgements

- Subject matter experts from AWS

## Introduction - Week 1

- This week focuses on
  - Transformer architecture.
  - Generative AI project lifecycle

## Generative AI & LLMs

- Lesson contents
  - Large language models
    - Their use cases
    - How the models work
    - Prompt engineering
    - How to make creative text outputs
  - Outline a project life cycle for generative AI projects

- The foundation models, with billions of parameters, exhibit emergent properties beyond language alone.
- Lab assignments would use Flan-T5 LLM.

## LLM use cases and tasks

- Base concept behind a number of different capabilities, starting with a basic chatbot:
  - Next word prediction
- Language understanding stored within the parameters of the model:
  - processes
  - reasons
  - ultimately solves the tasks given

## Text generation before transformers

### Generating text with RNNs

- RNNs were limited by the amount of compute and memory needed to perform well at generative tasks.
- To successfully predict the next word, models need to see more than just the previous few words.
  - Models need to have an understanding of the whole sentence or even the whole document.

### Transformers

- Paper referred: Attention is All you need
- Pros of transformers
  - Scale efficiently
    - Scaled efficiently to use multi-core GPUs
  - Parallel process input data, amking use of mush larger training datasets
  - Attention to input meaning
    - Ability to learn to pay attention to the meaning of the words it's processing.

## Transformers architecture

### Self-attention

- Attention map
  - Attention weights between each word and every other word.
- Multi-headed self-attention
  - Multiple sets of self-attention weights or heads are learned in parallel independently of each other.
  - Intuition:
    - Each self-attention head will **learn** a **different aspects of language**.
    - Example:
      - One head may see the relationship between the people entities in our sentence.
      - Another head may focus on the activity of the sentence.
- Output of attention is passed to a feed forward network.

## Generating text with transformers

- Input words should be tokenized using the same tokenizer that was used to train the network.
- Encoder output is a deep representation of the **structure** and **meaning** of the input sequence.
  - Encoder inputs ("prompts") with contextual understanding and produces one vector per input token.
    - ?? TODO Check if that's the case or it produces a single vector for the entire input.
- Decoder accepts input tokens and generates new tokens.

- Architecture variations:
  - Encoder Only Models
    - Example model: BERT
    - Example usecase: Classification tasks such as sentiment analysis
  - Encoder Decoder Models
    - Used in sequence to sequence model where output sequence length is variable.
    - Example models: BART, T5
    - Example usecase: Machine translation
  - Decoder Only Models
    - Example models: GPT family of models, BLOOM, Jurassic, LLaMA

## Transformers: Attention is all you need

- The paper proposes a neural network architecture that replaces traditional recurrent neural networks (RNN) and convolutional neural networks (CNN) with an entirely attention-based mechanism.
- The Transformer model uses self-attention to compute representations of input sequences, which allow it to capture long-term dependencies and parallelize computation effectively.
- Each layer consists of two sub-layers:
  - Multi-head self-attention
  - Feed-forward neural network
    - Applies a point-wise fully connected layer to each position separately and identically.
- [arxiv url](https://arxiv.org/abs/1706.03762)

## Prompting and prompt engineering

- Terminology
  - Prompt: The text fed into the model
  - Inference: The act of generating text
  - Completion: The output text
  - Context window: The full amount of text or the memory that is available to use for the prompt
- In-context learning
  - Providing examples inside the context window

### In-context learning (ICL) - zero shot inference

- Zero-shot inference
  - Including input data within the prompt

### In-context learning (ICL) - one shot inference

- The prompt text starts with a completed example that demonstrates the tasks to be carried out to the model.
- Now smaller model has a better chance of understanding the task you're specifying and the format of the response that you want.

### In-context learning (ICL) - few shot inference

- What if model isn't performing well even after providing five or six examples
  - Fine-tune your model

### The significance of scale: language understanding

- Largest models vs smaller models
  - Largest models:
    - Surpisingly good at zero-shot inference
    - Successfully complete many tasks that they were not specifically trained to perform.
  - Smaller models:
    - Generally only good at a small number of tasks, typically those that are similar to the task that they were trained on.

## Generative configuration

### Generative configuration - Inference parameters

- Max new tokens
- Sample top K
- Sample top P
- Temperature

### Generative configuration - greedy vs random sampling

- Greedy decoding: The word/token with the highest probability is selected.
  - Works well for short generation
  - Susceptible to repeated words or repeated sequences of words
- Random (weighted) sampling
  - Select a token using a random-weighted strategy across the probabilities of all tokens.

### Generative configuration - top-k

- Select an output from the top-k results after applying random-weighted strategy using the probabilities.
- This method can help the model have some randomness while preventing the selection of highly improbable completion words.

### Generative configuration - top-p sampling

- Select an output using the random-weighted strategy with the top-ranked consecutive results by probability and with a cumulative probability <= p.

### Generative configuration - temperature

- This parameter influences the shape of the probability distribution that the model calculates for the next token.
  - Cooler temperature (< 1)
    - Strongly peaked probability distribution
  - Higher temperature (> 1)
    - Broader, flatter probability distribution
  - Temperature = 1
    - This will leave softmax function as default
    - Unaltered probability will be used

## Generative AI project lifecycle

- Project lifecycle
  - Scope
    - Define the use case
    - LLMs are capable of carrying out many tasks, but their abilities depend strongly on the size and architecture of the model.
    - You should think about what function the LLM will have in your specific application
      - Do we need model to carry out many different tasks
      - Or model needs to be good at only one specific task
  - Select
    - Choose an existing model or pretrain your own
  - Adapt and align model
    - Prompt engineering
    - Fine-tuning
    - Align with human feedback
    - Evaluate
      - Metrics and benchmarks
  - Application integration
    - Optimize and deploy model for inference
    - Augment model and build LLM-powered applications
      - Fundamental limitations of LLMs that can be difficult to overcome through training alone
        - Tendency to invent information when they don't know an answer
        - Their limited ability to carry out complex reasoning and mathematics.

## Introduction to AWS labs

## Lab 1 walkthrough

## Lab 1 - Generative AI Use Case: Summarize Dialogue

- [Notebook](../code/Lab_1_summarize_dialogue.ipynb)
