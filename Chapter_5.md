# Reinforcement Learning from Human Feedback

1. [Introduction - Week 3](#introduction---week-3)
2. [Aligning models with human values](#aligning-models-with-human-values)
3. [Reinforcement learning from human feedback (RLHF)](#reinforcement-learning-from-human-feedback-rlhf)
4. [RLHF: Obtaining feedback from humans](#rlhf-obtaining-feedback-from-humans)
5. [RLHF: Reward model](#rlhf-reward-model)
6. [RLHF: Fine-tuning with reinforcement learning](#rlhf-fine-tuning-with-reinforcement-learning)
7. [Proximal policy optimization](#proximal-policy-optimization)
8. [RLHF: Reward hacking](#rlhf-reward-hacking)
9. [KL divergence](#kl-divergence)
10. [Scaling human feedback](#scaling-human-feedback)
11. [Lab 3 walkthrough](#lab-3-walkthrough)
12. [Lab 3 - Fine-tune FLAN-T5 with reinforcement learning to generate more-positive summaries](#lab-3---fine-tune-flan-t5-with-reinforcement-learning-to-generate-more-positive-summaries)

## Introduction - Week 3

- Topics to be covered
  - RLHF
  - Using LLMs as a reasoning engine

## Aligning models with human values

- In the Generative AI project lifecycle
  - Align with human feedback
    - Comes under Adapt and align model

- HHH (Human values)
  - Helpfulness
  - Honesty
  - Harmlessness

## Reinforcement learning from human feedback (RLHF)

- Instruct fine-tuned LLM -> RLHF -> Human-aligned LLM

- Reinforcement Learning (RL)
  - A type of machine learning in which agent learns to make decisions related to a specific goal by taking actions in an environment, with the objective of maximizing some notion of a cumulative reward.

- Example described: Tic-Tac-Toe

### Reinforcement learning: fine-tune LLMs

- Reward model
  - Additional model to classify the outputs of the LLM and evaluate the degree of alignment with human preferences.
  - Start with a smaller number of human examples to train the secondary model by traditional supervised learning methods.
  - Once trained, use the reward model to assess the output of the LLM and assign a reward value, which in turn gets used to update the weights of the LLM and train a new human aligned version.
- Rollout
  - The sequences of actions and states.
  - Term used in the context of language modeling
  - Playout: Term used in classical reinforcement learning

## RLHF: Obtaining feedback from humans

- Collect human feedback
  - Define your model alignment criterion
    - e.g. helpfulness, toxicity
  - For the prompt-response sets that you just generated, obtain human feedback through labeler workforce
    - This process gets repeated for many prompt completion sets, building up a dataset that can be used to train the **reward model** that will ultimately carry out this task instead of the humans.
- Prepare labeled data for training
  - Convert rankings into pairwise training data for the reward model
    - $N \choose 2$ combinations where $N$ = Number of alternative completions per prompt
    - {$y_j$, $y_k$}: reorder the prompts so that the preferred option comes first.
  
## RLHF: Reward model

- Train reward model
  - Train model to predict preferred completion from {$y_j, y_k$} for prompt $x$
- Reward model is also usually a language model
  - e.g. BERT trained using supervised learning methods on the pairwise comparison data that was prepared from the human labelers assessment off the prompts.
- $loss = log(\sigma (r_j - r_k))$
  - where $r_j$, $r_k$ are corresponding rewards
- Use the reward model as a binary classifier to provide reward value for each prompt-completion pair.
- Reward value in RLHF = [logit](https://en.wikipedia.org/wiki/Logit) value for the true class

## RLHF: Fine-tuning with reinforcement learning

- You want to start with a model that already has good performance on your task of interests.
- Stopping criterion for iterative process
  - Reaching a threshold value on some evaluation criteria
  - Maximum number of steps

## Proximal policy optimization

- Instructor: [Dr. Ehsan Kamalinejad](https://www.linkedin.com/in/ehsan-kamalinejad/)
  - AI researcher at Amazon
- Updates to the LLM
  - Small updates within a bounded region
    - This is where the proximal term comes from.
- Start PPO with initial instruct LLM
- At high level, each cycle of PPO goes over two phases
  - Phase 1:
    - The LLM is used to carry out a number of experiments, completing the given prompts.
    - Experiments assess the outcome of the current model
    - These experiments allows to update the LLM against  the reward model in Phase II.
    - Calculate value loss
      - Value loss: The difference between the actual future total reward and its approximation to the value function.
  - Phase 2:
    - Small updates are made to the model and the impact of the updates on the alignment goal for the model is evaluated.
    - Objective: To find a policy whose expected reward is high.
    - Loss equation
      - [OpenAI documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html#key-equations)
        - This documentation focus on the PPO-Clip variant.
        - For the PPO-Penalty variant, follow the [Hugging Face article](#kl-divergence) on Transformer Reinforcement Learning
      - ${\pi}_\theta$: Model's probability distribution over tokens
        - Probability of the next token with the LLM given the current prompt $s_t$
          - $s_t$: completed prompt up to the token $t$
      - $\hat{A}_t$: Estimated advantage term of a given choice of action
        - Estimates how much better or worse the current action is compared to all possible actions at that state.
      - Guardrails: Keeping the policy in the "trust region"
      - OpenAI documentation explains the clip term better for both the cases where advantage is positive/negative.
    - Entropy loss
      - $L^{ENT} = entropy(\pi_{\theta}(.|s_t))$
      - Entropy value:
        - Low: High probability of completing the prompt in the same way
        - High: Guides the LLM towards more creativity
      - Similar to temperature setting seen in Week 1
        - Difference
          - Temperature influences model creativity at the inference time
          - Entropy influences model creativity at the training time
    - Objective function
      - $L^{PPO} = L^{POLICY} + c_1*L^{VF} + c_2*l^{ENT}$
        - 1st term: Policy loss
        - 2nd term: Value loss
        - 3rd term: Entropy loss

## RLHF: Reward hacking

- As the policy tries to optimize the reward, it can diverge too much from the initial language model.
- Avoiding reward hacking
  - Initial instruct model can be used as performance reference, we can call it reference model.
  - Weights of the reference model are frozen and not updated during iterations of RLHF.
  - During training each prompt is passed to both models to generate completion
    - reference LLM
    - intermediate LLM updated model
  - Kullback-Leibler divergence can be used to compare the completions of the two models and determine how much the updated model has diverged from the reference.
  - KL divergence penalty gets added to reward
  - Combining with PEFT
    - Only update the weights of the PEFT adapter, not the full weights of the LLM
    - This means same underlying LLM can be used for both the reference model and the PPO model

## KL divergence

- A mathematical measure of the difference between two probability distributions, which helps us understand how one distribution differs from another.
- [Transformer Reinforcement Learning](https://huggingface.co/blog/trl-peft)
  - The training scripts seem to have moved to another urls.

## Scaling human feedback

- As the number of models and use cases increases, human effort becomes a limited resource.
- One idea to overcome these limitations is to scale through model self supervision.
- Constitutional AI
  - Model is trained in two distinct phases
    - Stage 1: Supervised learning
      - Start to prompt the model that try to get it to generate harmful responses (This process is called red teaming)
      - Ask the model to critique its own harmful responses according to the constitutional principles and revise them to comply with those rules.
      - Explained with the usecase of hacking someone's wifi
      - To mitigate harmful response, augment the prompt using the harmful completion and a set of predefined instructions that ask the model to critique its response.
      - using the rules outlined in the Constitution, the model detects the problems in its reponse.
      - Put all the parts together and ask the model to write a new response that removes all of the harmful or illegal content.
      - The original read team prompt, and the final constitutional response can then be used as training data to create a fine-tuned LLM that has learned to generate constitutional responses.
    - Stage 2: Reinforcement learning
      - Referred as Reinforcement learning from AI feedback (RLAIF)

## Lab 3 walkthrough

## Lab 3 - Fine-tune FLAN-T5 with reinforcement learning to generate more-positive summaries

- [Notebook](../code/Lab_3_fine_tune_model_to_detoxify_summaries.ipynb)
