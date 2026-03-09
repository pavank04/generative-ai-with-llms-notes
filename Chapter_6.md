# LLM-powered applications

1. [Model optimizations for deployment](#model-optimizations-for-deployment)
2. [Generative AI Project Lifecycle Cheat Sheet](#generative-ai-project-lifecycle-cheat-sheet)
3. [Using the LLM in applications](#using-the-llm-in-applications)
4. [Interacting with external applications](#interacting-with-external-applications)
5. [Helping LLMs reason and plan with chain-of-thought](#helping-llms-reason-and-plan-with-chain-of-thought)
6. [Program-aided language models (PAL)](#program-aided-language-models-pal)
7. [ReAct: Combining reasoning and action](#react-combining-reasoning-and-action)
8. [LLM application architectures](#llm-application-architectures)
9. [AWS SageMaker JumpStart](#aws-sagemaker-jumpstart)
10. [Week 3 Quiz](#week-3-quiz)
11. [Week 3 resources](#week-3-resources)

## Model optimizations for deployment

- Application integration block of Generative AI project lifecycle raises the following set of questions
  - Optimize and deploy model for inference
    - How your LLM will function in deployment
    - How fast your model should generate completions
    - What compute budget do you have available
    - Trade off model performance for improved inference speed or lower storage
  - Augment model and build LLM-powered applications
    - Questions tied to additional resources that your model may need
    - Do you intend your model to interact with external data or other applications
    - If so, how will you connect to those resources
    - How your model will be consumed
      - Intended application or API interface

- Inference challenges of LLMs are in terms of
  - Computing
  - Storage
  - Ensuring low latency for consuming applications

- LLM optimization techniques
  - Distillation
  - Quantization
  - Pruning

- Distillation
  - Train a smaller student model from a larger teacher model
  - Typically useful for encoder model and not for decoder model.
  - Triple loss function
    - Distillation loss
    - Supervised training loss
    - Cosine embedding loss
  - External resource (KA)
    - A [blog on Knowledge Distillation of Language Models](https://alexnim.com/coding-projects-knowledge-distillation.html)
      - Softmax temperature function
        - Shows with diagram how probability distribution changes with increase of temperature (T)
      - Figure explains the three losses

- Post-Training Quantization (PTQ)
  - Reduce precision of model weights
  - Applied to model weights (and/or activations)
  - Requires calibration
    - To statistically capture the dynamic range of the original parameter values

- Pruning
  - Remove model weights with values close or equal to zero
  - Pruning methods
    - Full model re-training
    - PEFT/LoRA
    - Post-training

## Generative AI Project Lifecycle Cheat Sheet

- Describes the expected time taken and technical expertise required for various tasks.

## Using the LLM in applications

- Difficulty of models
  - Out of date
  - Wrong (inability to do mathematical reasoning)
  - Hallucination

- Augment model and build LLM-powered applications
  - This section will teach techniques to help LLM overcome the above issues by connecting to external data sources and applications.
- Orchestration library
  - This layer augments and enhances the performance of the LLM at runtime by providing access to external data sources or connecting to existing APIs of other applications.
  - Example: Langchain
- Retrieval augmented generation (RAG)
  - Overcome knowledge cutoffs to give model access to additional external data at inference time.
  - The prompt is passed to the query encoder
    - Encodes the data in the same format as the external documents.
  - Searches for a relevant entry in the corpus of documents.
  - The retriever combines the new text with the original prompt.
  - The expanded prompt contains the specific case of interest and is then passed to the LLM.
  - The model uses the information in the context of the prompt to generate a completion that contains the correct answer.

- Vector store
  - Data storage strategy
  - Contains vector representations of text

- Two considerations for using external data in RAG
  - Data must fit inside context window
    - Split long sources into short chunks
  - Data must be in format that allows its relevance to be assessed at inference time: Embedding vectors

## Interacting with external applications

- Requirements for using LLMs to power applications
  - Plan actions
    - The model needs to be able to generate a set of instructions so that the application knows what actions to take.
  - Format outputs
    - The completions needs to be formatted in a way that the broader application can understand.
    - Could be as simple as specific sentence structure
    - Or as complex as writing a script in Python or generating a SQL command.
  - Validate actions
    - Collect required user information and make sure it is in the completion
- Prompt structure is important in all the above requirements.

## Helping LLMs reason and plan with chain-of-thought

- LLMs can struggle with complex reasoning problems
- Humans take a step-by-step approach to solving complex problems
  - Asking the LLM model to mimic reasoning steps human take is called "Chain of thought".
- Chain-of-Thought prompting can help LLMs reason
  - The one shot prompt now has the intermediate steps in place of just the answer.

## Program-aided language models (PAL)

- Strategy
  - LLM generates completions where reasoning steps are accompanied by computer code.
    - The output format of the model is specified for the model by including examples for one or few shot inference in the prompt.
  - This code is then passed to an interpreter to carry put the calculations necessary to solve the problem.

## ReAct: Combining reasoning and action

- ReAct: Synergizing Reasoning and Action in LLMs
  - Datasets utilized in this 2022 paper:
    - HotPot QA: multi-step question answering
    - Fever: Fact verification

- ReAct instructions define the action space
  - Solve a question answering task with interleaving Thought, Action, Observation steps
  - Thought can reason about the current situation, and Action can be three types
    - Search [entity]
    - Lookup [keyword]
    - Finish [answer]

- LangChain
  - The framework provides modular pieces that contain the components necessary to work with LLMs.
  - Combines into a "chain"
    - Prompt templates
    - Memory
    - Tools
    - Agents

## ReAct: Reasoning and action

- Paper: [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [github page](https://react-lm.github.io/)

## LLM application architectures

- In this final section, we'll explore some additional considerations for building LLM powered applications.

- Key components to create end-to-end solutions for LLM powered applications
  - Infrastructure
    - Training/Fine-Tuning, Serving, Application Components
    - Taking into account whether you need real-time or near-real-time interaction with the model.
    - Also have the need to retrieve information from external sources, such as [RAG](#using-the-llm-in-applications)
  - Information sources
    - Documents
    - Database
    - Web
  - LLM models
    - Foundations models or
    - Optimized LLM fine-tuned for specific task
  - LLM Tools and Frameworks e.g. LangChain, Model Hubs
  - Application Interfaces e.g. Websites, RestAPIs, Mobile Applications

## AWS SageMaker JumpStart

- SageMaker JumpStart is a ModelHub.

## Week 3 Quiz

## Week 3 resources

- Reinforcement Learning from Human-Feedback (RLHF)
  - [Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155.pdf)
  - [Learning to summarize from human feedback](https://arxiv.org/pdf/2009.01325.pdf)

- Proximal Policy Optimization (PPO)
  - [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)
  - [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/pdf/2305.18290.pdf)

- Scaling human feedback
  - [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/pdf/2212.08073.pdf)

- Advanced Prompting Techniques
  - [Chain-of-thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf)
  - [PAL: Program-aided Language Models](https://arxiv.org/abs/2211.10435)
  - [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

- LLM powered application architectures
  - [LangChain Library (GitHub)](https://github.com/hwchase17/langchain)
  - [Who Owns the Generative AI Platform?](https://a16z.com/2023/01/19/who-owns-the-generative-ai-platform/)
