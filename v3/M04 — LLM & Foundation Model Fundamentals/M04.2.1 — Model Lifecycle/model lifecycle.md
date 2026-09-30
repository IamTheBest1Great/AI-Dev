# M04.2.1 — Model Lifecycle

Think of an AI model like a **student**.

First, the student learns a huge amount of general knowledge → **pre-training**.
Then you teach the student how to follow instructions → **supervised post-training**.
Then you teach the student which answers humans prefer → **preference/reward-based post-training**.
Then you can make a smaller student learn from a larger student → **distillation**.
Finally, you prepare the student for the real world → **deployment**.

## The complete picture

```text
                 MODEL LIFECYCLE

Raw Internet / Books / Code / Data
              ↓
       1. PRE-TRAINING
              ↓
     Base / Foundation Model
              ↓
    2. SUPERVISED POST-TRAINING
       ├── Instruction Tuning
       └── Supervised Fine-Tuning
              ↓
     3. PREFERENCE POST-TRAINING
       ├── Preference Optimization
       ├── RLHF
       └── DPO
              ↓
      4. KNOWLEDGE TRANSFER
       ├── Distillation
       └── Synthetic Data
              ↓
       5. DEPLOYMENT
       ├── Quantization
       ├── Packaging
       ├── Serving
       ├── Versioning
       └── Evaluation
              ↓
          USERS / APPS
```

---

# 1. Training Stages

---

## M04.2.1.1.1 Pre-training

### What is pre-training?

**Pre-training is where the model learns general patterns from a huge amount of data.**

For an LLM, this could include:

* Books
* Websites
* Articles
* Code
* Documents
* Other text

The model sees enormous amounts of text and learns patterns such as:

```text
"The sun rises in the ___"
```

It learns that:

```text
east
```

is a likely continuation.

But don't think of it as simply memorizing sentences.

It gradually learns things such as:

* grammar
* vocabulary
* relationships between words
* coding patterns
* facts and concepts
* reasoning patterns
* how language is structured

### Simple example

Suppose the model sees:

```text
The cat drank the ___.
```

It learns that words like:

```text
milk
water
```

are likely continuations.

After seeing billions/trillions of examples, it builds a huge set of parameters that represent what it has learned.

### Main idea

> **Pre-training teaches the model about the world and language.**

### Important

The output of pre-training is usually called a:

**Base model** or **Foundation model**

---

# 2. Supervised Post-Training

After pre-training, the model may know a lot, but it may not behave like a helpful assistant.

For example, you ask:

```text
Explain photosynthesis in simple words.
```

A base model might continue the text unpredictably rather than acting like a polished assistant.

So we train it further.

---

# M04.2.1.1.2.1 Instruction Tuning

### What is instruction tuning?

We give the model examples of:

```text
Instruction → Good response
```

For example:

```text
User:
Translate "Hello" into Hindi.

Assistant:
नमस्ते
```

Another:

```text
User:
Write a Python function to add two numbers.

Assistant:
def add(a, b):
    return a + b
```

The model learns:

> "When a user gives me an instruction, I should respond appropriately."

### Before instruction tuning

The model is more like:

> "I understand language."

### After instruction tuning

The model becomes more like:

> "I understand language **and I know how to follow instructions.**"

### Easy memory trick

**Instruction tuning = Teach the model to follow instructions.**

---

# M04.2.1.1.2.2 Supervised Fine-Tuning (SFT)

### What is SFT?

SFT means **Supervised Fine-Tuning**.

You take a pre-trained model and train it further using a **smaller, carefully selected dataset**.

That dataset contains examples of desired behavior.

For example, suppose you are building a medical assistant.

You could have examples like:

```text
Question → Expert-approved answer
```

```text
What is hypertension?
→ Hypertension is persistently high blood pressure...
```

The model learns to produce responses similar to those examples.

### Why is it called "supervised"?

Because we provide the model with the **correct/desired answer**.

It's similar to a teacher checking homework.

```text
Student question
      ↓
Correct answer provided
      ↓
Model learns from it
```

### Important relationship

Instruction tuning is often performed using supervised fine-tuning.

So you will often see:

```text
Instruction Tuning ≈ a type/use of SFT
```

They are related, not completely separate concepts.

### Easy memory trick

**SFT = Show the model good examples and make it learn from them.**

---

# 3. Preference and Reward-Based Post-Training

Now we have a model that can follow instructions.

But another question appears:

> **Which of two acceptable answers is better?**

For example:

### Answer A

> Photosynthesis is the process by which plants convert light energy into chemical energy.

### Answer B

> Plants use sunlight, water and carbon dioxide to make food and release oxygen.

Both can be correct.

But depending on the user, **B may be easier to understand**.

So we introduce **preferences**.

---

# M04.2.1.1.3.1 Preference Optimization

### What is preference optimization?

Instead of only saying:

> "This is the correct answer."

we might say:

> "Answer B is better than Answer A."

Example:

```text
Question
   ↓
Answer A
Answer B
   ↓
Human preference
   ↓
B is preferred
```

The model learns which types of responses are preferred.

Preferences can involve things like:

* helpfulness
* correctness
* clarity
* harmlessness
* following instructions
* style

### Simple idea

**SFT:**
"Here is a good answer."

**Preference training:**
"Between these answers, this one is better."

---

# M04.2.1.1.3.2 RLHF

## RLHF = Reinforcement Learning from Human Feedback

This sounds complicated, but the basic idea is simple.

Humans look at model responses and provide feedback.

Example:

```text
Question:
Explain gravity.

Response A: ⭐⭐⭐⭐⭐
Response B: ⭐⭐⭐
```

The training system uses this preference information to encourage the model toward responses humans prefer.

### The basic flow

```text
Model generates answers
        ↓
Humans evaluate them
        ↓
Feedback / preferences
        ↓
Reward signal
        ↓
Model is optimized
```

### Why "reinforcement"?

Because the model is encouraged toward behavior that receives a higher reward.

Very simplified:

```text
Good behavior → higher reward
Bad behavior  → lower reward
```

### Important

RLHF isn't simply:

> "Humans manually write every answer."

Instead, humans provide feedback that helps shape the model's behavior.

### Memory trick

**RLHF = Humans give feedback, training uses that feedback as reward.**

---

# M04.2.1.1.3.3 DPO

## DPO = Direct Preference Optimization

DPO also uses **preference data**, but it is a different optimization approach from the traditional RLHF pipeline.

Suppose we have:

```text
Question:
Explain recursion.

Preferred answer:
Answer A ✅

Less preferred answer:
Answer B ❌
```

DPO learns from this pair.

The important idea is:

> **DPO directly trains the model using preferred vs non-preferred responses.**

You can think of it as:

```text
Preferred answer   ✅
Rejected answer    ❌
        ↓
       DPO
        ↓
Model learns the preference
```

### RLHF vs DPO — beginner view

| RLHF                                                           | DPO                                  |
| -------------------------------------------------------------- | ------------------------------------ |
| Uses human preference information                              | Uses human preference information    |
| Often involves a reward-model / RL-style optimization pipeline | Direct preference-based optimization |
| More involved pipeline                                         | Generally simpler training setup     |

For now, remember:

> **RLHF and DPO are two ways of using preference information to improve model behavior.**

---

# 4. Knowledge Transfer and Dataset Generation

Now we have another interesting problem.

Large models can be:

* expensive
* slow
* memory hungry

So we may want a smaller model.

---

# M04.2.1.1.4.1 Distillation

## What is distillation?

Imagine:

```text
Large Teacher
      ↓
Teaches
      ↓
Small Student
```

A large model is called the **teacher**.

A smaller model is called the **student**.

The student learns from the teacher's behavior.

### Example

You have:

```text
Teacher model = 70B parameters
Student model = 8B parameters
```

You want the 8B model to perform some useful tasks almost like the larger model.

The teacher can generate examples that the student learns from.

### Simple analogy

A highly experienced teacher teaches a younger student.

The student doesn't become identical to the teacher, but can learn many useful skills.

### Why use distillation?

Because smaller models can be:

* cheaper
* faster
* easier to deploy
* less demanding on hardware

### Memory trick

**Distillation = Big model teaches small model.**

---

# M04.2.1.1.4.2 Synthetic Data

## What is synthetic data?

**Synthetic data is data generated by another system, often an AI model, rather than collected directly from humans or the real world.**

Example:

You need 1 million training examples.

Instead of humans writing all 1 million examples:

```text
Large model
    ↓
Generates training examples
    ↓
Synthetic dataset
    ↓
Train another model
```

For example:

```text
Generate 100,000 math questions
Generate answers
Generate explanations
Generate coding examples
```

Then those examples can be filtered and used for training.

### Real data vs synthetic data

```text
Real data
→ written / observed from real sources

Synthetic data
→ generated artificially, often by another model
```

### Important

Synthetic data isn't automatically good.

It needs:

* quality checks
* filtering
* validation
* deduplication
* safety checks

Otherwise the model may learn bad or incorrect examples.

### Memory trick

**Synthetic data = Artificially generated training data.**

---

# 5. Deployment Stages

Training is finished.

Now comes:

> **How do we actually put the model into an application?**

This is deployment.

---

# M04.2.1.2.1 Quantization

## What is quantization?

A model contains huge numbers.

For example, its parameters may traditionally be stored using:

```text
FP32
```

or

```text
FP16 / BF16
```

Quantization means using **lower-precision representations**.

For example:

```text
16-bit → 8-bit
```

or sometimes:

```text
8-bit → 4-bit
```

### Why?

Lower precision can reduce:

* memory usage
* storage size
* inference cost

and may improve inference speed on compatible hardware.

### Example

Imagine:

```text
Original model
100 GB

Quantized model
50 GB
```

The exact reduction depends on the model and quantization method.

### Trade-off

Quantization can introduce some quality loss.

So:

```text
Smaller / faster
       ↕
Potential quality loss
```

### Memory trick

**Quantization = Make model numbers smaller to make the model cheaper/faster to run.**

---

# M04.2.1.2.2 Model Packaging

The trained model isn't necessarily a single convenient file ready for production.

You need to package things required to run it.

This may include:

```text
Model weights
+
Tokenizer
+
Configuration
+
Runtime dependencies
+
Inference settings
```

Think of it like preparing a product for shipping.

### Analogy

Training:

> Build the car.

Packaging:

> Put the car, manual, keys, and required equipment together for delivery.

### Memory trick

**Packaging = Prepare everything needed to run the model.**

---

# M04.2.1.2.3 Serving

## What is serving?

Serving means:

> **Making the model available so applications/users can send requests to it.**

For example:

```text
User
 ↓
Your application
 ↓
API request
 ↓
Model server
 ↓
LLM
 ↓
Response
```

Example:

```http
POST /generate
```

Your backend sends:

```text
"Explain Docker in simple words"
```

The model server runs inference and returns the answer.

### Popular serving concepts

You may later encounter things like:

* model servers
* GPUs
* batching
* throughput
* latency
* autoscaling
* inference endpoints

### Memory trick

**Serving = Put the model behind an interface so applications can use it.**

---

# M04.2.1.2.4 Versioning

Models change over time.

Suppose you deploy:

```text
Model v1
```

Then you improve it:

```text
Model v2
```

Then:

```text
Model v3
```

You need to know:

* which model is currently deployed
* what changed
* which dataset was used
* which configuration was used
* whether performance improved
* how to roll back

Example:

```text
customer-support-model:v1
customer-support-model:v2
customer-support-model:v3
```

### Why important?

Suppose v3 starts producing worse answers.

You need to know:

> "What exactly changed?"

And potentially return to v2.

### Memory trick

**Versioning = Keep track of different model releases.**

---

# M04.2.1.2.5 Evaluation Before Rollout

Before giving a model to millions of users, you should test it.

You ask questions such as:

```text
Is it accurate?
Is it following instructions?
Is it fast enough?
Does it hallucinate?
Is it safe?
Does it work on our actual use cases?
```

### Example

You have:

```text
Current model → v1
New model     → v2
```

Test both on an evaluation dataset.

```text
                 v1       v2
Accuracy        88%      91%
Latency         500ms    450ms
Safety score    ...      ...
```

You don't simply deploy v2 because it is newer.

You evaluate it against the requirements.

### Common evaluation categories

**Quality**

* correctness
* instruction following
* reasoning/task performance

**Safety**

* harmful outputs
* policy violations
* unwanted behavior

**Performance**

* latency
* throughput
* memory usage

**Business**

* cost
* task completion
* user experience

### Memory trick

**Evaluation = Test the model before exposing it to real users.**

---

# Putting Everything Together

Here's the most important diagram to remember:

```text
                 ┌───────────────────┐
                 │   RAW DATA        │
                 │ books/web/code    │
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │   PRE-TRAINING    │
                 │ learn language    │
                 │ + general patterns│
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │ INSTRUCTION / SFT │
                 │ learn to follow   │
                 │ instructions      │
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │ PREFERENCE TRAIN  │
                 │ RLHF / DPO        │
                 │ learn preferences │
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │ DISTILLATION      │
                 │ + SYNTHETIC DATA  │
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │    QUANTIZATION   │
                 │ smaller/faster    │
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │     PACKAGING     │
                 │ prepare runtime   │
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │      SERVING      │
                 │ API / inference   │
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │    EVALUATION     │
                 │ test before users │
                 └─────────┬─────────┘
                           ↓
                 ┌───────────────────┐
                 │      USERS        │
                 └───────────────────┘
```

# The easiest way to remember the whole lifecycle

Think:

## **Learn → Teach → Prefer → Transfer → Prepare → Run → Check**

| Stage                       | Simple meaning                                         |
| --------------------------- | ------------------------------------------------------ |
| **Pre-training**            | Learn general knowledge/patterns                       |
| **Instruction tuning**      | Learn to follow instructions                           |
| **SFT**                     | Learn from good example answers                        |
| **Preference optimization** | Learn which answers are preferred                      |
| **RLHF**                    | Use human feedback in a reward-based training approach |
| **DPO**                     | Directly learn from preferred vs rejected responses    |
| **Distillation**            | Big model teaches small model                          |
| **Synthetic data**          | AI-generated training examples                         |
| **Quantization**            | Reduce numerical precision to save resources           |
| **Packaging**               | Bundle everything required to run the model            |
| **Serving**                 | Make the model available through an application/API    |
| **Versioning**              | Track model releases and changes                       |
| **Evaluation**              | Test before rollout                                    |

---

# One Very Important Distinction

As a beginner, don't mix these three ideas:

### Training

```text
Change the model's parameters
```

Examples:

* Pre-training
* SFT
* Preference optimization
* Distillation

### Data generation

```text
Create examples used for training
```

Example:

* Synthetic data

### Deployment

```text
Prepare and run the trained model
```

Examples:

* Quantization
* Packaging
* Serving
* Versioning
* Evaluation

---

# A Real-World Example

Imagine you're building a **customer-support LLM**.

### Step 1 — Pre-training

Train on a huge amount of general text.

```text
Internet + books + code + documents
             ↓
       Base model
```

### Step 2 — SFT

Give it customer-support examples:

```text
Customer question
        ↓
Good support answer
```

Now it learns the desired style.

### Step 3 — Preference training

Give it two possible support answers:

```text
Answer A
Answer B
```

Human reviewers indicate which response is preferable.

The model learns those preferences.

### Step 4 — Synthetic data

Ask a stronger model to generate thousands of customer-support scenarios.

```text
Large model
    ↓
Synthetic examples
    ↓
Training
```

### Step 5 — Distillation

Create a smaller model that learns from the larger model.

```text
Large model
     ↓
   teaches
     ↓
Small model
```

### Step 6 — Quantization

Convert the model to a lower-precision format to reduce serving cost.

### Step 7 — Packaging

Bundle:

```text
weights
tokenizer
config
runtime
```

### Step 8 — Serving

Put it behind an API:

```text
React app
   ↓
Backend
   ↓
Model API
   ↓
LLM
```

### Step 9 — Versioning

Track:

```text
support-model-v1
support-model-v2
support-model-v3
```

### Step 10 — Evaluation

Test the new model before rollout.

```text
Accuracy ✅
Latency ✅
Cost ✅
Safety ✅
Customer tasks ✅
```

Then deploy it.

---

# 🧠 Final Mental Model

Remember this sentence:

> **Pre-training teaches the model to understand patterns, SFT teaches it how to follow examples and instructions, preference training teaches it what responses are preferred, distillation transfers capability to smaller models, and deployment prepares the model to run efficiently for real users.**

And the simplest lifecycle is:

```text
DATA
 ↓
LEARN
 ↓
FOLLOW INSTRUCTIONS
 ↓
LEARN PREFERENCES
 ↓
TRANSFER / IMPROVE
 ↓
OPTIMIZE FOR DEPLOYMENT
 ↓
SERVE
 ↓
EVALUATE
 ↓
USERS
```

This is the foundation you need before moving into more advanced topics such as **RLHF reward models, PPO, rejection sampling, LoRA/PEFT, model merging, quantization formats, batching, KV cache, inference servers, and evaluation frameworks**.
