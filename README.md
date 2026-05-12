# ML_Practice

## TL;DR

A personal machine learning and quantitative finance practice repository showcasing projects/tutorials across **Machine Learning, predictive modeling, NLP, recommender systems, deep learning, transformers, multi-agent AI workflows, and financial engineering.**

### Core Focus Areas:

- **Modern LLM Stack**: Transformers (from scratch & fine-tuning), LangChain, LangGraph, ReAct agents, retrieval-augmented agents, and multi-agent orchestration. NLP projects/tutorials covering topic modeling, clustering, summarization, multilingual NER, and transformer internals.
- **Quantitative Finance**: Yield curve modeling, stock prediction, PCA/SVD on ETFs, alternative data, option pricing, regime detection and time-series forecasting.
- **Applied ML**: Sports prediction, recommender systems, topic modeling, time series forecasting and deep learning projects.
- **Research Replication**: Critical analysis and reproduction of academic papers on stock prediction using optimized technical indicators and neural networks

**Tech Stack**: Python, PyTorch, TensorFlow, Hugging Face, LangChain/LangGraph, Haystack, scikit-learn, pandas, NumPy, SQL, Quantitative Finance tools.

## 1. [Predictive Analysis of Football Match Outcomes](/projects/project-01-predictive_analysis_of_football_match_outcomes.ipynb)

This project explores the application of machine learning in sports analytics, specifically focusing on binary classification of football match outcomes. It demonstrates data analysis, feature engineering (including time-series features and team performance metrics), and the evaluation of various ML models (Random Forests, Neural Networks, etc.). The project showcases statistical analysis, probability theory application, and the challenges of predicting inherently random events. It includes model validation techniques and performance analysis using metrics such as accuracy, precision, and recall.

## 2. [Recommendation/](/recommendation/)

This repository contains a Recommender System built on the concepts of collaborative filtering. The system can be used to provide personalized recommendations based on user ratings and preferences.

## 3. [Topic Modeling and Document Clustering - Aug 2024](https://colab.research.google.com/drive/1PdtqQ_nfgiLU56QD8FxXIBj0ZCifF784?usp=sharing)

This notebook focuses on document clustering and topic modeling, aiming to analyze, categorize large sets of text data and extract latent topics. It includes techniques for dimensionality reduction, clustering, evaluating the quality of clusters, and uses bertopic for topic modeling and representations.

## 4. [Rock Paper Scissors Player](https://github.com/The-Professor99/rock-paper-scissors-project)

An algorithm designed to play a game of Rock-Paper-Scissors (RPS) using traditional statistical analysis and sequence modeling.

## 5. [Linear Regression Health Costs Calculator - Mar 2024](https://colab.research.google.com/drive/1Gm8rj6VTBbcSKPZzB2LFGHo3VnfeK0Uj?usp=sharing)

A deep learning model trained to predict healthcare costs.
[FreeCodeCamp Challenge](https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/linear-regression-health-costs-calculator)

## 6. [Cat and Dog Image Classifier](https://colab.research.google.com/drive/1JBmMUJukeqTt5X4zLyEod9g75LYaZ-wm?usp=sharing)

A deep learning model trained to classify images of dogs and cats.
[FreeCodeCamp Challenge](https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/cat-and-dog-image-classifier)

## 7. [Neural Network SMS Text Classifier - Mar 2024](https://colab.research.google.com/drive/10AOuGvD-M8-ROxuKrLXQAnVPXNyY-Bs9?usp=sharing)

A neural network model trained to predict if an email is spam or ham
[FreeCodeCamp Challenge](https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/neural-network-sms-text-classifier)

## 8. [Dialogue Summarizer - Aug 2024](https://colab.research.google.com/drive/1F1kSGEmDSSieuUVpBKTcm6r9ehpRJPTm?usp=sharing)

This notebook demonstrates the fine-tuning of a transformer model(google/pegasus-cnn_dailymail) on the SAMSum dataset to summarize chat dialogues. The process involves data preprocessing, training the model on conversational data, and evaluating the summarization results using metrics such as ROUGE. Note that this was done for practice purposes, so the fine-tuning process was not fully optimized for production use.

## 9. [DSN Hackathon - Aug 2023](hackathons/dsn)

[Data Science Nigeria Hackathon](https://zindi.africa/competitions/free-ai-classes-in-every-city-hackathon-2023) notebooks centered on predicting house prices of houses in Nigeria. The notebooks focus on data cleaning, exploratory analysis, and modeling pipelines.

**Tech Stack**: pandas, EDA, regression, feature engineering

## 10. [Hands-on Learning: Transformers, Agents, Multi Agents](/hands_on_learning/transformers_agents_multiagents)

A set of tutorial notebooks exploring agent orchestration and transformer workflows with LangChain, LangGraph, and Hugging Face tools.

- **[Advanced Multi-Agent Workflow with LangGraph - May 2026](https://colab.research.google.com/drive/1txUROkx_Tv9baBcVFRLYRhqlw4xrb5N2?usp=sharing)**

  Explores a stateful multi-agent architecture built with LangGraph, including a supervisor/orchestrator pattern, specialized subagents, human-in-the-loop debugging, and structured performance evaluation. The notebook focuses on reducing hallucinations, managing conversation flow, and instrumenting the agent pipeline with LangSmith-style observability.

  **Tech Stack**: LangGraph, LangChain, LangSmith concepts, OpenAI-compatible chat models

  **Key Learning**: hierarchical agent design, stateful workflows, multi-agent coordination, and debugging/metrics for agentic applications.

- **LangChain Chatbot Prototyping**

  **Notebooks:**
  [Notebook 1](/hands_on_learning/transformers_agents_multiagents/Chatbot.ipynb), [Notebook 2](/hands_on_learning/transformers_agents_multiagents/Chatbot_Copy1.ipynb), [Notebook 3](/hands_on_learning/transformers_agents_multiagents/Chatbot_2.ipynb)

  Hands-on chatbot experiments that demonstrate prompt design, model definition, and agent-based conversation flow. These notebooks serve as practical prototypes for building conversational assistants with LangChain.

  **Key Learning**: agent-driven chatbots, prompt engineering, conversational tool usage.

- **[Text Clustering and Topic Discovery](/hands_on_learning/transformers_agents_multiagents/KMeans_Text_Clustering.ipynb)**

  Implements document clustering using Bag-of-Words, TF-IDF/Hashing feature extraction, LSA dimensionality reduction, and KMeans clustering. Includes cluster evaluation with homogeneity, completeness, V-measure, Adjusted Rand Index, and silhouette analysis.

  **Tech Stack**: scikit-learn, KMeans, TF-IDF, HashingVectorizer, LSA

  **Key Learning**: text clustering pipelines, sparse data clustering, and quality metrics for unsupervised NLP.

- **[LangChain Fundamentals and Deployment Concepts](/hands_on_learning/transformers_agents_multiagents/Langchain.ipynb)**

  Covers core LangChain concepts such as chains, prompts, output parsers, ChatModel vs LLM, LangSmith debugging, and LangServe deployment. Also compares open-source model usage with managed OpenAI workflows.

  **Key Learning**: chain construction, prompt templates, message roles, and LangChain deployment patterns.

- **[Agent Retrieval and Tool Integration](/hands_on_learning/transformers_agents_multiagents/Langchain_Learn.ipynb)**

  Demonstrates how to build retrieval-enabled agents, convert retrievers into tools, and let agents manage retrieval decisions. Includes examples of when an agent does or does not execute retrieval based on query intent.

  **Key Learning**: tool-enabled agents, conversational retrieval, and retrieval-augmented generation behavior.

- **[Conversational Memory and Agent Refresher](/hands_on_learning/transformers_agents_multiagents/Langchain_Refresher.ipynb)**

  Reviews message history, stateful conversation handling, session IDs, and how to wrap language models with memory for chat applications.

  **Key Learning**: session history, stateful chat workflows, and memory-backed agent patterns.

- **[Transformer Architecture from Scratch](https://colab.research.google.com/drive/1F0QRNf3bqOEr4PnPuTDqVaFwa92fM5I8?usp=sharing)**

  Builds transformer components in PyTorch, including embeddings, positional encodings, attention score computation, and the core self-attention mechanism.

  **Tech Stack**: PyTorch, transformer internals, model implementation

  **Key Learning**: how token embeddings and self-attention operate at the architectural level.

- **[Text Summarization Workflow](https://colab.research.google.com/drive/1ZS0KoNgq9sWdoaSoC2azK4SfuPulhvfW?usp=sharing)**

  Explores summarization of long articles with transformer models, handling context limits, data preparation, and evaluation heuristics such as newline-separated summaries and sentence boundary handling.

  **Key Learning**: summarization dataset prep, long-context limitations, and baseline model understanding.

- **[Multilingual NER and Transfer Learning](https://colab.research.google.com/drive/1Vr1A3H0gBpaDAY08_OUc2mkVdfGT_d3h?usp=sharing)**

  Works on named entity recognition using PAN-X data, creating a Swiss-like multilingual corpus with German, French, Italian, and English text. Focuses on zero-shot transfer and cross-lingual evaluation for LOC/PER/ORG labels.

  **Key Learning**: multilingual NER, dataset sampling, and cross-lingual model evaluation.

- **[Hugging Face Training Refresher](https://colab.research.google.com/drive/1H2rb6dEnjrX5Y6qbgt2HC6u4Cx9qyo9I?usp=sharing)**

  Reviews both native Python training loops and the Hugging Face Trainer API for fine-tuning transformer models.

  **Key Learning**: training loop structure, Trainer usage, and model fine-tuning patterns.

**Tech Stack**: LangChain, LangGraph, Hugging Face Transformers, PyTorch, NLP, KMeans

**Key Learning**: multi-agent workflows, chatbot prototyping, transformer fundamentals, and retrieval-augmented agent patterns.

## 11. WorldQuant Program Group Projects (Apr 2025 - Present)

Some of the group work projects I participated in at WorldQuant University involved collaborative code implementations and the preparation of comprehensive final reports as the primary deliverables. These projects focused on applied quantitative finance skills. They include, but are not limited to:

- **Data Quality and Yield Curve Modeling**

  Assessed structured and unstructured financial data quality, then modeled government bond term structures using Nelson-Siegel and cubic spline approaches. The project compared model fit across short- and long-term maturities, evaluated goodness-of-fit, analyzed parameter interpretations, and discussed the ethical implications of smoothing yield curves in financial reporting.

  **Tech Stack**: fixed income modeling, term structure analysis, model comparison

  **Key Learning**: data quality evaluation, yield curve fitting, model interpretation, parameter estimation, ethical considerations in quantitative finance, and financial data engineering.

- **Empirical Analysis of ETF**

  Performed an in-depth analysis of Exchange-Traded Funds (ETFs), applying transformations and presenting results through clear, concise reports.

  **Tech Stack and Key Learnings**: PCA, SVD, covariance matrix analysis, dimensionality reduction in portfolio context, financial returns calculation, in-depth mathematical interpretation.

- **Stock Market Prediction in Emerging Markets (Paper Implementation)**:

  Conducted an in-depth academic review and practical replication of the paper “An Intelligent Approach for Predicting Stock Market Movements in Emerging Markets Using Optimized Technical Indicators and Neural Networks” (Sagaceta Mejia et al.). Implemented optimized indicator-based feature engineering with neural networks to forecast stock price movements, achieving performance consistent with published results.

- **Option Pricing**

  Validated vanilla option prices using binomial and trinomial tree models, applied put-call parity checks, compared European and American option valuations, and computed Greeks for risk assessment. Also assessed option pricing using Heston and Merton frameworks, with Monte Carlo simulation for volatility and jump dynamics. The project analyzed pricing under different correlation regimes and calculated Greeks.

  **Tech Stack**: options pricing, tree-based valuation, risk sensitivities, stochastic volatility, jump-diffusion models, Monte Carlo simulation

  **Key Learning**: pricing model validation, parity relationships, and Greek computation, advanced option modeling, volatility regime analysis, and risk metric computation.

- **Omitted Variable Bias and Specification**

  Analyzed regression misspecification by comparing models with and without omitted variables, using simulation to demonstrate the effects on parameter estimates and inference.

  **Tech Stack**: econometric diagnostics, simulation-based analysis, regression modeling

  **Key Learning**: omitted variable bias, model specification, and parameter interpretation.

- **Time Series and Feature Engineering Challenges**

  Addressed time series challenges including feature extraction, non-stationarity, equilibrium modeling, multicollinearity, and regime detection. The emphasis was on selecting appropriate models and explaining their financial implications.

  **Tech Stack**: time-series econometrics, feature engineering, regime identification

  **Key Learning**: stationarity handling, feature selection, and regime-switching interpretation.

- **Quantitative Model Survey**

  Created a structured survey of finance-focused ML techniques, covering regularized linear models, clustering, PCA, tree-based methods, discriminant analysis, support vector machines, and neural networks, comparing advantages, limitations, and application scenarios.

  **Tech Stack**: regression, clustering, dimensionality reduction, tree models, SVM, neural networks

  **Key Learning**: model selection, regularization, comparative evaluation, and financial ML strategy.

- **Hyperparameter Tuning and Generalization**

  Focused on validation, hyperparameter optimization, and the bias-variance tradeoff to improve out-of-sample performance. The project also explored ways to combine models for more robust results.

  **Tech Stack**: model tuning, validation analysis, ensemble reasoning

  **Key Learning**: bias-variance tradeoff, hyperparameter optimization, and model generalization.

- **Regime-Based Allocation Strategy**

  Designed a volatility-regime allocation framework with Markov chains and Hidden Markov Models. The project covered data preparation, regime identification, and rule-based allocation logic.

  **Tech Stack**: regime switching, Markov chains, HMMs, asset allocation

  **Key Learning**: volatility regime modeling, regime-driven allocation, and time-series strategy design.

## 12. [Hands-on Learning: Tinkerings](hands_on_learning/tinkerings)

A pair of practical notebooks focused on exploratory forecasting and foundational dimensionality reduction.

- `001_Time_Series_Stores_Sales_Forecasting_EDA0.ipynb`
  **Retail Sales Forecasting EDA**

  Analyzes store-level retail sales data with features like store number, product family, promotion flags, holidays, oil prices, and transaction counts. Covers lag creation, dataset preparation, and business-context insights for time-series forecasting.

  **Tech Stack**: pandas, time-series feature engineering, EDA, ARIMA

  **Key Learning**: forecasting data preparation, lag engineering, and holiday/event impact analysis.

- `dim_reduction.ipynb`
  **Dimensionality Reduction and Linear Algebra Deep Dive**

  A conceptual tutorial on eigenvalues, eigenvectors, SVD, PCA, and their relationships to collaborative filtering and dimensionality reduction. Includes mathematical explanations and practical intuition.

  **Tech Stack**: linear algebra, PCA, SVD, matrix decompositions

  **Key Learning**: eigen decomposition, low-dimensional projections, and the math behind feature reduction.

## 13. [Hands-on Learning: Aurelien Geron Hands-On ML with Scikit-Learn](/hands_on_learning/hands_on_ml_with_scikit-learn_Aurelien_textbook)

A collection of tutorials based on Aurelien Geron's Hands-On Machine Learning with Scikit-Learn. These notebooks follow the book's chapters and provide practical exercises for supervised learning, model evaluation, ensemble methods, and end-to-end ML pipelines.

**Scope**: supervised learning, validation, feature engineering, ensemble methods, and neural network workflows.

**Key Learning**: Scikit-Learn best practices, applied book-based tutorials, model validation strategies, and end-to-end machine learning development.
