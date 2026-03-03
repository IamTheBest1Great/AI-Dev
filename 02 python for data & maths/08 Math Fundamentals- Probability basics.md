# Day 14: Probability Basics – The Language of Uncertainty

Welcome to Day 14! Probability is the mathematical foundation for dealing with uncertainty, and it's everywhere in AI:
- **Machine learning models** make predictions with uncertainty.
- **Bayesian inference** updates beliefs based on data.
- **Reinforcement learning** balances exploration and exploitation.
- **Natural language processing** uses probabilistic language models.
- **Computer vision** often involves probabilistic interpretations.

Today we'll cover:
- Fundamental concepts: sample space, events, probability axioms.
- Conditional probability and independence.
- Bayes' theorem – a cornerstone of AI.
- Random variables and probability distributions (discrete and continuous).
- Expectation and variance (review with probability context).
- Sampling and empirical distributions.
- Key distributions: Bernoulli, Binomial, Poisson, Uniform, Normal (Gaussian).
- Law of Large Numbers and Central Limit Theorem.

Let's build your probabilistic intuition with Python examples.

---

## 1. What is Probability?

Probability measures the likelihood of an event occurring. It ranges from 0 (impossible) to 1 (certain).

### Basic Terminology
- **Experiment**: A process with uncertain outcomes (e.g., rolling a die).
- **Sample Space (Ω)**: Set of all possible outcomes (e.g., {1,2,3,4,5,6}).
- **Event**: A subset of the sample space (e.g., "rolling an even number" = {2,4,6}).
- **Probability P(E)**: A number assigned to an event satisfying certain axioms.

### Axioms of Probability (Kolmogorov)
1. \(0 \leq P(E) \leq 1\) for any event E.
2. \(P(\Omega) = 1\) (the whole sample space has probability 1).
3. For mutually exclusive events \(E_1, E_2, \ldots\), \(P(\bigcup_i E_i) = \sum_i P(E_i)\).

---

## 2. Simple Probability with Python

We can simulate experiments and compute empirical probabilities.

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulate rolling a fair die 10000 times
rolls = np.random.randint(1, 7, size=10000)

# Probability of rolling a 6
prob_6 = np.sum(rolls == 6) / len(rolls)
print(f"Empirical probability of rolling a 6: {prob_6:.4f}")
print(f"Theoretical probability: {1/6:.4f}")

# Probability of an even number
prob_even = np.sum(rolls % 2 == 0) / len(rolls)
print(f"Empirical probability of even: {prob_even:.4f}")
print(f"Theoretical probability: {3/6:.4f}")
```

---

## 3. Conditional Probability and Independence

**Conditional probability** \(P(A|B)\) is the probability of event A given that B has occurred:

\[
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0
\]

**Independence**: Events A and B are independent if

\[
P(A \cap B) = P(A) \cdot P(B) \quad \text{or} \quad P(A|B) = P(A)
\]

### Example: Drawing cards
What's the probability of drawing a king given that the card is a face card?

```python
# Simulate drawing cards from a deck
# Deck: 52 cards, 4 suits, 13 ranks. Face cards: Jack, Queen, King (12 total)
# Kings: 4

# Empirical approach
n_sim = 100000
# Simulate drawing a card: 1-52, but let's use ranks 1-13 and suits for simplicity
# Instead, we'll just compute theoretically.
# Theoretical: P(King | Face) = (4/52) / (12/52) = 4/12 = 1/3

# But let's simulate
suits = np.repeat(['H','D','C','S'], 13)
ranks = np.tile(np.arange(1,14), 4)  # 1= Ace, 11=J,12=Q,13=K
deck = list(zip(ranks, suits))

# Draw one card
draws = np.random.choice(len(deck), size=n_sim, replace=True)
is_face = (ranks[draws] >= 11)  # J,Q,K
is_king = (ranks[draws] == 13)

# Conditional probability: P(king | face)
if np.sum(is_face) > 0:
    prob_king_given_face = np.sum(is_king & is_face) / np.sum(is_face)
    print(f"Empirical P(King | Face): {prob_king_given_face:.4f}")
print(f"Theoretical: {1/3:.4f}")
```

---

## 4. Bayes' Theorem

Bayes' theorem relates conditional probabilities:

\[
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
\]

In machine learning, this is the foundation of **Bayesian inference** and the **Naive Bayes classifier**.

### Example: Medical Test
A disease affects 1% of the population. A test is 99% accurate (true positive rate 99%, false positive rate 1%). If a person tests positive, what's the probability they actually have the disease?

```python
# P(disease) = 0.01
# P(positive|disease) = 0.99
# P(positive|no disease) = 0.01

p_disease = 0.01
p_pos_given_disease = 0.99
p_pos_given_healthy = 0.01

# P(positive) = P(positive|disease)*P(disease) + P(positive|healthy)*P(healthy)
p_pos = p_pos_given_disease * p_disease + p_pos_given_healthy * (1 - p_disease)

# P(disease|positive)
p_disease_given_pos = (p_pos_given_disease * p_disease) / p_pos
print(f"Probability of having disease given positive test: {p_disease_given_pos:.4f}")
# Output: ~0.5 (counterintuitive, but due to low base rate)
```

---

## 5. Random Variables

A **random variable** is a variable whose possible values are outcomes of a random phenomenon. It can be **discrete** (taking countable values) or **continuous** (taking any value in an interval).

- **Discrete**: e.g., number of heads in 10 coin flips.
- **Continuous**: e.g., height of a randomly selected person.

### Probability Mass Function (PMF) for discrete
\(P(X = x)\) gives the probability that X takes value x.

### Probability Density Function (PDF) for continuous
\(f(x)\) such that \(P(a \le X \le b) = \int_a^b f(x) dx\).

### Cumulative Distribution Function (CDF)
\(F(x) = P(X \le x)\) for both discrete and continuous.

---

## 6. Common Probability Distributions

### 6.1 Discrete Distributions

#### Bernoulli Distribution
Models a single trial with two outcomes (success/failure), e.g., a coin flip. Parameter \(p\) = probability of success.

```python
from scipy.stats import bernoulli

p = 0.3
# PMF at x=1
prob_1 = bernoulli.pmf(1, p)
print(f"P(X=1) = {prob_1}")

# Generate 10 samples
samples = bernoulli.rvs(p, size=10)
print(samples)
```

#### Binomial Distribution
Number of successes in \(n\) independent Bernoulli trials. Parameters: \(n\), \(p\).

```python
from scipy.stats import binom

n, p = 10, 0.5
# PMF for exactly 6 heads
prob_6 = binom.pmf(6, n, p)
print(f"P(6 heads in 10 flips) = {prob_6:.4f}")

# CDF: probability of <= 6 heads
cdf_6 = binom.cdf(6, n, p)
print(f"P(<=6 heads) = {cdf_6:.4f}")

# Generate samples
samples = binom.rvs(n, p, size=1000)
plt.hist(samples, bins=np.arange(-0.5, n+1.5, 1), density=True, alpha=0.6)
plt.title('Binomial Distribution (n=10, p=0.5)')
plt.show()
```

#### Poisson Distribution
Models number of events in a fixed interval of time/space, with average rate \(\lambda\). Often used in count data.

```python
from scipy.stats import poisson

lam = 3
# Probability of exactly 2 events
prob_2 = poisson.pmf(2, lam)
print(f"P(X=2) with λ=3: {prob_2:.4f}")

# Generate samples
samples = poisson.rvs(lam, size=1000)
plt.hist(samples, bins=np.arange(-0.5, max(samples)+1.5, 1), density=True, alpha=0.6)
plt.title('Poisson Distribution (λ=3)')
plt.show()
```

### 6.2 Continuous Distributions

#### Uniform Distribution
All values in an interval \([a,b]\) equally likely.

```python
from scipy.stats import uniform

a, b = 0, 1
# PDF at 0.5
pdf_val = uniform.pdf(0.5, loc=a, scale=b-a)
print(f"PDF at 0.5: {pdf_val}")  # 1.0

# Generate samples
samples = uniform.rvs(loc=a, scale=b-a, size=1000)
plt.hist(samples, bins=30, density=True, alpha=0.6)
x = np.linspace(a, b, 100)
plt.plot(x, uniform.pdf(x, loc=a, scale=b-a), 'r-', label='PDF')
plt.title('Uniform Distribution [0,1]')
plt.legend()
plt.show()
```

#### Normal (Gaussian) Distribution
The bell curve, characterized by mean \(\mu\) and standard deviation \(\sigma\). Central in statistics due to Central Limit Theorem.

```python
from scipy.stats import norm

mu, sigma = 0, 1
# PDF at x=0
pdf_val = norm.pdf(0, loc=mu, scale=sigma)
print(f"PDF at 0: {pdf_val:.4f}")  # ~0.3989

# CDF at x=1.96
cdf_val = norm.cdf(1.96, loc=mu, scale=sigma)
print(f"P(X <= 1.96): {cdf_val:.4f}")  # ~0.975

# Generate samples
samples = norm.rvs(loc=mu, scale=sigma, size=1000)
plt.hist(samples, bins=30, density=True, alpha=0.6)
x = np.linspace(-4, 4, 100)
plt.plot(x, norm.pdf(x, loc=mu, scale=sigma), 'r-', label='PDF')
plt.title('Standard Normal Distribution')
plt.legend()
plt.show()
```

---

## 7. Expectation and Variance

For a random variable \(X\):

- **Expectation (mean)**: \(E[X] = \sum x P(X=x)\) (discrete) or \(\int x f(x) dx\) (continuous). It's the long-run average.
- **Variance**: \(Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2\). Measures spread.

We can compute these theoretically for distributions or empirically from samples.

```python
# For a binomial distribution with n=10, p=0.5
n, p = 10, 0.5
mean_theor = n * p
var_theor = n * p * (1-p)
print(f"Theoretical mean: {mean_theor}, variance: {var_theor}")

# Empirical from samples
samples = binom.rvs(n, p, size=10000)
mean_emp = np.mean(samples)
var_emp = np.var(samples, ddof=1)  # sample variance
print(f"Empirical mean: {mean_emp:.2f}, variance: {var_emp:.2f}")
```

---

## 8. Law of Large Numbers and Central Limit Theorem

### Law of Large Numbers
As the number of trials increases, the empirical average converges to the expected value.

```python
# Roll a die: theoretical mean = 3.5
means = []
n_trials = np.arange(1, 10000, 10)
for n in n_trials:
    rolls = np.random.randint(1, 7, size=n)
    means.append(np.mean(rolls))

plt.plot(n_trials, means)
plt.axhline(y=3.5, color='r', linestyle='--', label='Theoretical mean')
plt.xlabel('Number of rolls')
plt.ylabel('Empirical mean')
plt.title('Law of Large Numbers')
plt.legend()
plt.show()
```

### Central Limit Theorem
The sum (or mean) of a large number of independent random variables tends to a normal distribution, regardless of the original distribution.

```python
# Take samples from a uniform distribution, compute means of samples
sample_means = []
for _ in range(1000):
    sample = uniform.rvs(size=30)  # 30 samples from Uniform
    sample_means.append(np.mean(sample))

plt.hist(sample_means, bins=30, density=True, alpha=0.6)
x = np.linspace(0.3, 0.7, 100)
plt.plot(x, norm.pdf(x, loc=0.5, scale=np.sqrt(1/12/30)), 'r-')  # CLT predicts normal with mean 0.5, var (1/12)/30
plt.title('Distribution of Sample Means (n=30) from Uniform[0,1]')
plt.show()
```

---

## 9. Probability in AI/ML – Where You'll Use It

- **Naive Bayes Classifier**: Based on Bayes' theorem, assumes conditional independence.
- **Loss Functions**: Mean Squared Error is related to variance; cross-entropy derives from probability theory.
- **Generative Models**: Like GANs and VAEs, model the probability distribution of data.
- **Reinforcement Learning**: Policies are probability distributions over actions.
- **Bayesian Neural Networks**: Learn distributions over weights.
- **Evaluation Metrics**: Precision, recall, F1 are based on conditional probabilities.

---

## 10. Summary

| Concept | Formula / Key Idea | Python Tools |
|---------|--------------------|--------------|
| Probability of event | \(P(E) = \frac{\text{favorable}}{\text{total}}\) | `np.mean(condition)` |
| Conditional probability | \(P(A|B) = P(A∩B)/P(B)\) | Conditional filtering |
| Bayes' theorem | \(P(A|B) = P(B|A)P(A)/P(B)\) | Use with given probabilities |
| Random variable | Maps outcomes to numbers | `scipy.stats` distributions |
| Expectation | \(E[X] = \sum x P(x)\) | `np.mean(samples)` |
| Variance | \(Var(X) = E[(X-μ)^2]\) | `np.var(samples, ddof=1)` |
| Discrete distributions | Bernoulli, Binomial, Poisson | `scipy.stats.bernoulli`, etc. |
| Continuous distributions | Uniform, Normal | `scipy.stats.uniform`, `norm` |

---

## 11. Practice Tasks

1. **Coin Flips Simulation**
   - Simulate flipping a fair coin 1000 times. Compute the empirical probability of heads.
   - Now simulate 1000 flips of a biased coin with p=0.3. Compute the probability of heads.

2. **Conditional Probability**
   - Create a dataset of 1000 random people with two attributes: smoker (yes/no) and lung disease (yes/no). Define probabilities arbitrarily (e.g., P(smoker)=0.2, P(disease|smoker)=0.1, P(disease|non-smoker)=0.01). Generate the data and compute empirical P(disease|smoker) and compare to theoretical.

3. **Bayes' Theorem in Action**
   - Use the medical test example but change the numbers. Compute posterior probability for a test with 95% accuracy and disease prevalence 5%. Verify with a simulation (generate a population, apply test, compute proportion of positive tests that actually have disease).

4. **Exploring Distributions**
   - Generate 1000 samples from a Poisson distribution with λ=5. Plot the histogram and overlay the theoretical PMF.
   - Generate 1000 samples from a Normal distribution with μ=10, σ=2. Compute empirical mean and variance, and compare with theoretical.

5. **Central Limit Theorem Demo**
   - Choose a non-normal distribution (e.g., exponential with scale=1). Take 500 samples of size 30, compute means, and plot histogram. Observe it looks normal.

6. **Mini Project: Naive Bayes from Scratch**
   - Using the Iris dataset, implement a simple Naive Bayes classifier manually (using probability estimates) for one feature (e.g., sepal length) to predict species. Compute conditional probabilities using empirical distributions (e.g., assume Gaussian and estimate μ, σ per class). Test on a few samples.

---

You've now built a solid foundation in probability – a crucial pillar of AI. Next, we could explore **linear algebra review** or dive directly into **machine learning algorithms**. Let me know your preference!
