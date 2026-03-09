# LLM pre-training and scaling laws

1. [Pre-training large language models](#pre-training-large-language-models)
2. [Computational challenges of training LLMs](#computational-challenges-of-training-llms)
3. [Efficient Multi-GPU Compute Strategies](#efficient-multi-gpu-compute-strategies)
4. [Scaling laws and compute-optimal models](#scaling-laws-and-compute-optimal-models)
5. [Pre-training for domain adaptation](#pre-training-for-domain-adaptation)
6. [Domain-specific training: BloombergGPT](#domain-specific-training-bloomberggpt)
7. [Week 1 resources](#week-1-resources)

## Pre-training large language models

### Considerations for choosing a model

- Options
  - Existing model: Foundation model
  - Train your own from scratch: Custom model

### Model hubs

- Variance of the transformer model architecture are suited to different language tasks, largely because of differences in how the models are trained.

### Model architectures and pre-training objectives

- LLMs encode a deep statistical representation of language.
- In the self-supervised learning step, the model internalizes the patterns and structures present in the language.

### LLM pre-training at a high level

- The training objective depends on the model architecture.

### Autoencoding models

- Pretrained using Masked Language Modeling (MLM)
- Objective: Reconstruct text ("denoising")
- Good use cases:
  - Sentiment analysis
  - Named entity recognition
  - Word classification
- Examples:
  - BERT
  - ROBERTA

### Autoregressive models

- Decoder only models
- Pretrained using Causal Language Modeling (CLM)
- Objective: Predict next token
- Unidirectional context
- By learning to predict the next token from a vast number of examples, the model builds up a statistical representation of language.
- Larger decoder-only models show strong zero-shot inference abilities, and can often perform a range of tasks well.
- Good use cases:
  - Text generation
- Example models:
  - GPT
  - BLOOM

### Sequence-to-sequence models

- Encoder-Decoder LLM
- Exact details of the pre-training objective vary from model to model.
- T5
  - Pre-trains the encoder using span corruption, which masks random sequences of input tokens.
  - Objective of decoder: Reconstruct span
- Good use cases:
  - Translation
  - Text summarization
  - Question answering
- Example models
  - T5
  - BART

### The significance of scale: task ability

### Model size vs time

- Growth powered by
  - Introduction of massively scalable transformer architecture
  - Access to massive datasets

## Computational challenges of training LLMs

### Approximate GPU RAM needed to store 1B parameters

- 1 parameter = 4 bytes (32-bit float)
- 1B parameter = $4 * 10^9$ bytes = 4 GB

### Additional GPU RAM needed to train 1B parameters

- [Anatomy of Model's Memory](https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#anatomy-of-models-memory)
  - Components on GPU memory
    - model weights
    - optimizer states
    - gradients
    - forward activations saved for gradient computation
    - temporary buffers
    - functionality specific memory
  - HuggingFace documentation describe the details.

### Approximate GPU RAM needed to train 1B params

- Memory needed to store model
  - 4 GB @ 32-bit full precision
- Memory needed to train model
  - 24 GB @ 32-bit full precision

### Quantization

- 32-bit floating point => 16-bit/8-bit floating point
- Corresponding data types used in deep learning frameworks
  - FP32
    - Default for model weights, activations, and other model parameters.
  - FP16, Bfloat16 for 16-bit half precision, int8
    - BF16
      - Brain Floating Point format developed at Google Brain
      - Hybrid between half precision FP16 and full precision FP32

- Binary representation
  - FP32
    - 1st bit: Sign (0 represents positive)
    - Next 8 bits: Exponent
    - Next 23 bits: Represents fraction
      - Also referred as Mantissa/Significand=Precision
  - FP16
    - 1st bit: Sign
    - Next 5 bits: Exponent
    - Next 10 bits: Fraction
  - BF16
    - 1st bit: Sign
    - Next 8 bits: Exponent
    - Next 7 bits: Fraction

- Quantization statistically projects the original 32-bit floating point numbers into a lower precision space, using scaling factors calculated based on the range of the original 32-bit floaitng point numbers.

### Quantization: Summary

- Reduce required memory to store and train models
- Project original 32-bit floaitng point numbers into lower precision spaces
- Quantization-aware training (QAT) learns the quantization scaling factors during training
- BFLOAT16 is a popular choice
  - Flan T5 trained using BF16

## Efficient Multi-GPU Compute Strategies

### Distributed Data Parallel (DDP)

- Copies model onto each GPU
- Sends batches of data to each of the GPUs in parallel.
- Each data-set is processed in parallel.
- A synchronization step combines the results of each GPU, which in turn updates the model on each GPU.

### Fully Sharded Data Parallel (FSDP)

- Motivated by the Zero paper - zero data overlap between GPUs
  - [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
  - [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)
  - Allows to scale model training across GPUs when model doesn't fit in the memory of a single chip.

### Zero Redundancy Optimizer (ZeRO)

- Reduces memory by distributing (sharding) the model parameters, gradients, and optimizer states across GPUs.
- Offers three optimization stages
  - Stage 1: Shards only optimizer states across GPUs
    - Reduces memory footprint by a factor of 4
  - Stage 2: Shards the gradients across chips
  - Stage 3: Shards all components including the model parameters across GPUs
    - When applied together with stages 1 and 2, memory reduction is linear with number of GPUs
- In contrast to GDP, where each GPU has all of the model states required for processing each batch of data available locally, FSDP requires you to collect this data from all of the GPUs before the forward and backward pass.

### Fully Sharded Data Parallel

- Helps to reduce overall GPU memory utilization
- Supports offloading to CPU if needed
- Configure level of sharding via sharding factor
  - To manage the trade-off between performance and memory utilization
- Full sharding
  - Most memory savings, but increases the communication volume between GPUs

## Scaling laws and compute-optimal models

### Scaling choices for pre=-training

- Scaling choice
  - Dataset size
  - Model size (number of parameters)
- Constraint
  - Compute budget

### Compute budget for training LLMs

- petaflop/s-day
  - #floating point operations performed at rate of 1 petaFLOP per second for one day
  - Equivalent to 8 NVIDIA V100s or 2 NVIDIA A100s running at full efficiency for 24 hours.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
  - Power law between test loss and compute

### Dataset size and model size vs performance

- Compute resource constraints
  - Hardware
  - Project timeline
  - Financial budget
- Power law observed between test loss and dataset size
  - Given compute budget and model size are held fixed.
- Power law observed between test loss and model size
  - Given compute budget and training dataset size are held constant.

### Chinchilla paper

- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)
- Observation
  - Very large models may be over-parameterized and under-trained.
    - Under-trained: Model would benefit from training over more data.
  - The authors hypothesize that smaller models trained on more data could perform as well as large models if they are trained on larger datasets.
  - Compute optimal training datasize is ~20x number of parameters.
    - LLaMA-65B: Found to be trained on compute optimal training datasize.
    - GPT-3, OPT-175B, BLOOM: Found to be under-trained.

## Pre-training for domain adaptation

- Explanation for the requirement of pre-training for specialized domain e.g. legal
  - Vocabulary specific to that domain which is rare in general language usage.
  - Common words used in the domain which have a different meaning to the one in general language usage.

### BloombergGPT: domain adaptation for finance

- Researchers combined finance and general datasets.
- Followed Chinchilla's scaling law.

## Domain-specific training: BloombergGPT

- BloombergGPT is a large decoder-only language model.
- [Paper](https://arxiv.org/abs/2303.17564)

## Week 1 resources

- Transformer Architecture
  - [Attention is All You Need](https://arxiv.org/pdf/1706.03762)
  - [BLOOM: BigScience 176B Model](https://arxiv.org/abs/2211.05100)
    - [High level overview](https://bigscience.notion.site/BLOOM-BigScience-176B-Model-ad073ca07cdf479398d5f95d88e218c4)
  - [Vector Space Models](https://www.coursera.org/learn/classification-vector-spaces-in-nlp/home/week/3)
    - Series of lessons from DeepLearning.AI's Natural Language Processing specialization
- Pre-training and scaling laws
  - [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- Model architectures and pre-training objectives
  - [What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?](https://arxiv.org/pdf/2204.05832.pdf)
  - HuggingFace
    - [Tasks](https://huggingface.co/tasks)
    - [Model Hub](https://huggingface.co/models)
  - [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/pdf/2302.13971.pdf)
- Scaling laws and compute-optimal models
  - [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
  - [Training Compute-Optimal Large Language Models](https://arxiv.org/pdf/2203.15556.pdf)
  - [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/pdf/2303.17564.pdf)
