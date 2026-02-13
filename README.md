# Lab 3: Contextual Bandit-Based News Article Recommendation

**Course:** Reinforcement Learning Fundamentals  
**Student:** Devyansh Choudhary  
**Roll Number:** U20230046  
**GitHub Branch:** Devyansh_U20230046

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Approach and Design Decisions](#approach-and-design-decisions)
- [Algorithms Implemented](#algorithms-implemented)
- [Key Results and Observations](#key-results-and-observations)
- [Requirements](#requirements)
- [Installation](#installation)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Project Structure](#project-structure)
- [References](#references)

---

## Overview

This project implements a **contextual bandit-based recommendation system** for news articles. The system learns to recommend articles from different categories (Entertainment, Education, Tech, Crime) to users with varying preferences by balancing exploration and exploitation using reinforcement learning techniques.

The implementation demonstrates:
- User context classification using machine learning
- Three bandit algorithms: ε-Greedy, Upper Confidence Bound (UCB), and SoftMax
- Hyperparameter sensitivity analysis
- Performance comparison across algorithms

---

## Problem Statement

The goal is to build a news recommendation system that:

1. **Classifies users** into different types based on their context (demographics, behavior, preferences)
2. **Recommends articles** from multiple categories to maximize user engagement (rewards)
3. **Balances exploration** (trying different articles) with **exploitation** (recommending known good articles)
4. **Adapts online** as it receives feedback from user interactions

This is modeled as a **contextual multi-armed bandit problem** where:
- **Arms** represent article category-context pairs (e.g., Tech articles for user_2)
- **Context** represents the classified user type
- **Reward** represents user engagement/satisfaction with the recommendation

---

## Approach and Design Decisions

### 1. **Data Generation and User Modeling**

The experiment uses synthetic data generated from the `rlcmab-sampler` package:
- **Three user types** (user_1, user_2, user_3) with distinct preference profiles
- **Four article categories** (Entertainment, Education, Tech, Crime)
- **12 total arms** (4 categories × 3 user types)
- Each user type has different reward distributions for different categories

### 2. **Context Classification**

A **Random Forest Classifier** is trained to predict user types from context features:
- **Purpose**: Enables the bandit to select appropriate arms based on classified user context
- **Performance Metric**: Classification accuracy
- **Result**: 89.75% accuracy on test set

Alternative classifiers tested:
- Decision Tree Classifier
- Logistic Regression
- Random Forest (selected for best performance)

### 3. **Bandit Algorithms**

Three classic bandit algorithms are implemented:

#### **ε-Greedy**
- **Strategy**: Explore with probability ε, exploit best known arm with probability (1-ε)
- **Hyperparameters tested**: ε ∈ {0.01, 0.1, 0.3}
- **Best configuration**: ε = 0.01

#### **Upper Confidence Bound (UCB)**
- **Strategy**: Select arms based on optimistic estimates using upper confidence bounds
- **Formula**: UCB(a) = Q(a) + C × √(ln(t) / N(a))
- **Hyperparameters tested**: C ∈ {0.5, 1.0, 2.0}
- **Best configuration**: C = 0.5

#### **SoftMax (Boltzmann Exploration)**
- **Strategy**: Probabilistic selection based on Q-values with temperature parameter
- **Formula**: P(a) ∝ exp(Q(a) / τ)
- **Hyperparameter tested**: τ = 1.0

### 4. **Experimental Setup**

- **Training episodes**: 1000
- **Learning rate**: Dynamic (1/N(a) for each arm)
- **Initialization**: Q-values initialized to 0
- **Evaluation metrics**: Cumulative reward, average reward, regret
- **Random seed**: 42 (for reproducibility)

---

## Algorithms Implemented

### ε-Greedy Algorithm

```
For each episode:
  1. Classify user context
  2. With probability ε: select random arm
     Otherwise: select arg max Q(a)
  3. Receive reward r
  4. Update: Q(a) ← Q(a) + α(r - Q(a))
```

**Characteristics**:
- Simple and computationally efficient
- Fixed exploration rate
- Performance depends heavily on ε value

### Upper Confidence Bound (UCB)

```
For each episode:
  1. Classify user context
  2. Calculate UCB for each arm:
     UCB(a) = Q(a) + C × √(ln(t) / N(a))
  3. Select arm with highest UCB
  4. Receive reward r
  5. Update Q(a) and N(a)
```

**Characteristics**:
- Optimistic exploration (confidence interval approach)
- Reduces exploration over time naturally
- Theoretically sound with logarithmic regret bounds

### SoftMax (Boltzmann)

```
For each episode:
  1. Classify user context
  2. Calculate probabilities:
     P(a) = exp(Q(a)/τ) / Σ exp(Q(i)/τ)
  3. Sample arm based on probabilities
  4. Receive reward r
  5. Update Q(a)
```

**Characteristics**:
- Smooth probabilistic exploration
- Temperature parameter controls randomness
- Can be sensitive to Q-value scales

---

## Key Results and Observations

### Algorithm Performance

| Algorithm | Hyperparameter | Average Reward | Rank |
|-----------|----------------|----------------|------|
| **UCB** | C = 0.5 | **6.7482** | 1st |
| **ε-Greedy** | ε = 0.01 | 6.5938 | 2nd |
| **SoftMax** | τ = 1.0 | 6.5343 | 3rd |

### Key Findings

1. **UCB outperforms other algorithms** with the lowest exploration parameter (C=0.5), achieving the highest average reward of 6.7482

2. **User Classification Accuracy**: 89.75% - High accuracy enables effective context-based arm selection

3. **Learned Q-values reveal clear preferences**:
   - **user_1**: Prefers Education (Q=3.10), followed by Crime (Q=1.65)
   - **user_2**: Strong preference for Tech (Q=8.94) and Education (Q=4.83)
   - **user_3**: Prefers Entertainment (Q=8.20) and Tech (Q=6.31)

4. **Hyperparameter Sensitivity**:
   - **ε-Greedy**: Lower ε values (0.01) performed best, suggesting rewards are well-separated
   - **UCB**: Lower C values (0.5) achieved faster convergence with sufficient exploration
   - **SoftMax**: Moderate temperature (τ=1.0) provided good balance

5. **Convergence Behavior**:
   - All algorithms show learning and convergence over 1000 episodes
   - UCB demonstrates most stable long-term performance
   - ε-Greedy shows faster initial learning but potentially suboptimal exploration

### Algorithm Trade-offs

| Aspect | ε-Greedy | UCB | SoftMax |
|--------|----------|-----|---------|
| Computational Complexity | O(1) | O(K) | O(K) |
| Exploration Strategy | Fixed random | Optimistic | Probabilistic |
| Hyperparameter Sensitivity | High | Medium | High |
| Theoretical Guarantees | None | Strong | Limited |
| Practical Performance | Good | **Best** | Good |
