# Festus Ihechi - Technical Portfolio

<a id="top"></a>

A results-driven ML engineer and frontend developer who builds production-ready AI systems, data pipelines, and user-facing web apps. I focus on end-to-end solutions, from data collection and modeling to deployment and monitoring, that deliver measurable product impact.

Current focus areas:

- Applied Machine Learning Engineering
- Time Series Forecasting and Quantitative Finance
- AI for Finance
- LLMs and Multi-Agent Systems
- Reinforcement Learning

[[View Resume](https://docs.google.com/document/d/1kXPjulitpBfVbe-aj-XAfvvYqVJzkNYpys4-2VdB_LY/edit?usp=sharing)]

**Featured Projects:**

**Top highlights:**

- [Yivera](#yivera) — music distribution platform frontend and admin dashboard.
- [Syftset Core Application](#syftset-core-application) — algorithmic trading operations, risk and portfolio management.
- [Recommendation System](#recommendation-system---jun-2024) — collaborative filtering + Streamlit deployment.
- [Rock Paper Scissors Player](#rock-paper-scissors-player---may-2024) — sequence modeling and statistical play prediction.
- [AWS Custom GAN](#aws-custom-gan---nov-2022) — AWS-based GAN and generative model experimentation.
- [Class Attendance Management System](#class-attendance-management-system---2021) — face recognition attendance automation.

**Additional featured work:**

- [Topic Modeling and Document Clustering](#topic-modeling-and-document-clustering---aug-2024) — document clustering & BERT-based topic extraction.
- [Building Machine Learning Models from Scratch](#building-machine-learning-models-from-scratch) — custom model implementations and fundamentals.
- [Regime-Based Allocation Strategy](#regime-based-allocation-strategy) — volatility regime detection and allocation logic.
- [Syftset v2](#syftset-v2) — trading dashboard with Firestore-backed analytics.
- [Syftset v1 Backend](#syftset-v1---backend---2022) — Flask-SocketIO trading signal backend.
- [Music Downloader](#music-downloader---dec-2020) — scraping automation for music retrieval.
- [Dialogue Summarizer](#dialogue-summarizer---aug-2024) — fine-tuned summarization on conversational data.

**Skills & Tools:**

- Languages: Python, JavaScript/TypeScript, SQL
- ML & Data: scikit-learn, TensorFlow, PyTorch, Hugging Face, pandas, NumPy
- NLP & Retrieval: transformers, sentence-transformers, bertopic, Haystack
- LLMs and Agent Systems: LangChain, LangGraph, LangSmith, Google's Agent Development Kit (ADK)
- Frontend: React, Next.js, MUI, Tailwind, SCSS, Angular
- Backend & Cloud: Flask, FastAPI, Firebase, AWS, GCP, Django
- Data Stores & Tools: PostgreSQL, Firestore, Elasticsearch, Docker, Git, CI/CD

# Projects

[Back to top](#top)

# Table of Contents

- [Machine Learning and AI](#machine-learning-and-ai)
- [WorldQuant Program Group Projects](#worldquant-program-group-projects)
- [Frontend Engineering](#frontend-engineering)
- [Backend Engineering](#backend-engineering)
- [Quantitative Finance and Trading Systems](#quantitative-finance-and-trading-systems)
- [Python Projects](#python-projects)
- [Other Projects](#other-projects)
- [Courses and Certifications](#courses-and-certifications)
- [Education](#education)
- [Hands on Learning](#hands-on-learning)
- [Contact](#contact)

---

# Machine Learning and AI

[Back to top](#top)

Some of my machine learning and AI projects. They include, but are not limited to:

- [Topic Modeling and Document Clustering - Aug 2024](#topic-modeling-and-document-clustering---aug-2024)
- [Dialogue Summarizer - Aug 2024](#dialogue-summarizer---aug-2024)
- [Recommendation System - Jun 2024](#recommendation-system---jun-2024)
- [Rock Paper Scissors Player - May 2024](#rock-paper-scissors-player---may-2024)
- [Cat and Dog Image Classifier - Apr 2024](#cat-and-dog-image-classifier---apr-2024)
- [Neural Network SMS Text Classifier - Mar 2024](#neural-network-sms-text-classifier---mar-2024)
- [Linear Regression Health Costs Calculator - Mar 2024](#linear-regression-health-costs-calculator---mar-2024)
- [Multilingual NLP and Question Answering Systems with Transformers and Haystack - Nov 2023](#multilingual-nlp-and-question-answering-systems-with-transformers-and-haystack---nov-2023)
- [Transformer Architecture from Scratch - Nov 2023](#transformer-architecture-from-scratch---nov-2023)
- [DSN Hackathon - Aug 2023](#dsn-hackathon---aug-2023)
- [Predictive Analysis of Football Match Outcomes - Jul 2023](#predictive-analysis-of-football-match-outcomes---jul-2023)
- [AWS Custom GAN - Nov 2022](#aws-custom-gan---nov-2022)
- [Class Attendance Management System - 2021](#class-attendance-management-system---2021)

## Topic Modeling and Document Clustering - Aug 2024

This notebook focuses on document clustering and topic modeling, aiming to analyze, categorize large sets of text data and extract latent topics. It includes techniques for dimensionality reduction, clustering, evaluating the quality of clusters, and uses bertopic for topic modeling and representations.

**Focus Areas**: topic modeling, document clustering

**Tech Stack**: `Python` `Numpy` `Pandas` `bertopic` `sentence-transformers` `nltk` `umap` `hdbscan` `wordcloud` `sklearn` `matplotlib`

**Links**

- Colab: [View Notebook](https://colab.research.google.com/drive/1PdtqQ_nfgiLU56QD8FxXIBj0ZCifF784?usp=sharing)

## Dialogue Summarizer - Aug 2024

This notebook demonstrates the fine-tuning of a transformer model(google/pegasus-cnn_dailymail) on the SAMSum dataset to summarize chat dialogues. The process involves data preprocessing, training the model on conversational data, and evaluating the summarization results using metrics such as ROUGE. Note that this was done for practice purposes, so the fine-tuning process was not fully optimized for production use.

**Focus Areas**: rouge evaluation, model fine-tuning

**Tech Stack**: `Python` `Numpy` `Pandas` `Huggingface datasets` `transformers` `pytorch` `nltk`

**Links**

- Colab: [View Notebook](https://colab.research.google.com/drive/1F1kSGEmDSSieuUVpBKTcm6r9ehpRJPTm?usp=sharing)

## Recommendation System - Jun 2024

This repository contains a Recommender System built on the concepts of collaborative filtering. The system can be used to provide personalized recommendations based on user ratings and preferences.

**Focus Areas**: recommendation system, collaborative filtering

**Tech Stack**: `Python` `Numpy` `Pandas` `sklearn` `streamlit` `scipy` `firebase`

**Links**

- Streamlit: [View Deployment](https://movie-recommendation-project.streamlit.app/)
- Github: [Repository Link](/recommendation/)

## Rock Paper Scissors Player - May 2024

An algorithm designed to play a game of Rock-Paper-Scissors (RPS) using traditional statistical analysis and sequence modeling.

**Focus Areas**: markov chains, markov decision processes

**Tech Stack**: `Python`

**Links**

- Replit: [View Deployment](https://replit.com/@festusihechi99/rpsgame)
- Github: [Repository Link](https://github.com/The-Professor99/rock-paper-scissors-project)

## Cat and Dog Image Classifier - Apr 2024

A deep learning model trained to classify images of dogs and cats.
[FreeCodeCamp Challenge](https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/cat-and-dog-image-classifier)

**Tech Stack**: `Python` `tensorflow` `numpy` `matplotlib`

**Links**

- Colab: [View Notebook](https://colab.research.google.com/drive/1JBmMUJukeqTt5X4zLyEod9g75LYaZ-wm?usp=sharing)

## Neural Network SMS Text Classifier - Mar 2024

A neural network model trained to predict if an email is spam or ham
[FreeCodeCamp Challenge](https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/neural-network-sms-text-classifier)

**Focus Areas**: exploratory data analysis (EDA), feature selection and extraction, classification

**Tech Stack**: `python` `tensorflow` `sklearn` `numpy` `pandas` `matplotlib`

**Links**

- Colab: [View Notebook](https://colab.research.google.com/drive/10AOuGvD-M8-ROxuKrLXQAnVPXNyY-Bs9?usp=sharing)

## Linear Regression Health Costs Calculator - Mar 2024

A deep learning model trained to predict healthcare costs.
[FreeCodeCamp Challenge](https://www.freecodecamp.org/learn/machine-learning-with-python/machine-learning-with-python-projects/linear-regression-health-costs-calculator)

**Focus Areas**: exploratory data analysis (EDA), feature selection and extraction, regression

**Tech Stack**: `python` `tensorflow` `sklearn` `numpy` `pandas` `matplotlib`

**Links**

- Colab: [View Notebook](https://colab.research.google.com/drive/1Gm8rj6VTBbcSKPZzB2LFGHo3VnfeK0Uj?usp=sharing)

### Multilingual NLP and Question Answering Systems with Transformers and Haystack - Nov 2023

Explored advanced natural language processing techniques focusing on multilingual Named Entity Recognition (NER), text generation strategies, and building an end-to-end Extractive Question Answering (QA) system. Implemented zero-shot cross-lingual transfer learning, custom token classification models, and a complete retriever-reader pipeline using sparse and dense retrieval methods.

Focus Areas: transformer models, named entity recognition, text generation (greedy & beam search), extractive question answering, retriever-reader architecture, domain adaptation, model evaluation

**Tech Stack**: `Python` `Transformers` `Huggingface Datasets` `Haystack` `PyTorch` `Pandas` `Numpy` `Elasticsearch` `FARM`

**Key Learning**: multilingual datasets, zero-shot cross-lingual transfer, custom token classification models, text generation decoding strategies, QA retriever-reader pipelines, BM25 and DPR retrieval, domain adaptation, model evaluation with recall, EM and F1 scores

**Links**

- Colab: [View Notebook](https://colab.research.google.com/drive/1Vr1A3H0gBpaDAY08_OUc2mkVdfGT_d3h?usp=sharing)

### Transformer Architecture from Scratch - Nov 2023

Builds transformer components in PyTorch, including embeddings, positional encodings, attention score computation, and the core self-attention mechanism.

**Focus Areas**: implementing the transformer architecture from scratch.

**Tech Stack**: `transformers` `pytorch`

**Links**

- Colab: [View Notebook](https://colab.research.google.com/drive/1F0QRNf3bqOEr4PnPuTDqVaFwa92fM5I8?usp=sharing)

## DSN Hackathon - Aug 2023

[Data Science Nigeria Hackathon](https://zindi.africa/competitions/free-ai-classes-in-every-city-hackathon-2023) notebooks centered on predicting house prices of houses in Nigeria. The notebooks focus on data cleaning, exploratory analysis, and modeling pipelines.

**Focus Areas**: exploratory data analysis (EDA), feature engineering, regression analysis

**Tech Stack**: `python` `sklearn` `numpy` `pandas` `matplotlib` `tensorflow` `keras`

**Links**

- Github: [Repository Link](hackathons/dsn)

## Predictive Analysis of Football Match Outcomes - Jul 2023

This project explores the application of machine learning in sports analytics, specifically focusing on binary classification of football match outcomes. It demonstrates data analysis, feature engineering (including time-series features and team performance metrics), and the evaluation of various ML models (Random Forests, Neural Networks, etc.). The project showcases statistical analysis, probability theory application, and the challenges of predicting inherently random events. It includes model validation techniques and performance analysis using metrics such as accuracy, precision, and recall.

**Focus Areas**: exploratory data analysis (EDA), feature engineering, regression analysis, modeling pipeline, learning and validation curves, hyperparameter tuning

**Tech Stack**: `python` `sklearn` `numpy` `pandas` `matplotlib` `category_encoders` `catboost` `tensorflow` `keras`

**Links**

- Github: [Repository Link](/projects/project-01-predictive_analysis_of_football_match_outcomes.ipynb)

## AWS Custom GAN - Nov 2022

The project uses AWS services, including AWS DeepComposer, AWS SageMaker, Amazon DynamoDB, AWS Lambda and AWS Step Funcitons, to build and train a custom generative adversarial network (GAN) for music generation and evaluation

**Focus Areas**: Machine Learning with AWS, Generator and Discriminator architecture, ML techniques and Generative AI, GANs, pre-trained deep learning models, u-net architecture

**Tech Stack**: `Python` `numpy` `scipy` `matplotlib` `seaborn` `tensorflow` `AWS` `Amazon DynamoDB` `AWS DeepComposer` `AWS SageMaker`

## Class Attendance Management System - 2021

This is a class attendance management system based on face recognition technology and was my final year project. It leverages facial recognition to automate student enrollment, identification, and attendance tracking. The system features a complete machine learning pipeline covering biometric enrollment, facial embedding generation, SVM-based classification, and attendance record management through an intuitive PyQt6 dashboard, allowing lecturers to manage attendance records through a graphical interface.

**Focus Areas**: class attendance management, facial recognition, data collection, system design, computer vision, machine learning, biometric systems, desktop application development, pre-trained deep learning models

**Tech Stack**: `python` `sklearn` `numpy` `pandas` `pyqt6` `postgresql` `opencv` `caffe models`

**Key Learning**: face detection and recognition pipelines, transfer learning, feature embedding extraction, Support Vector Machines (SVMs), biometric authentication systems, confidence threshold tuning, dataset preprocessing, desktop GUI development, attendance management workflows

**Links**

- Github: [Repository Link](https://github.com/The-Professor99/CAMSyFReT)
- Pypi: [View on Pypi](https://pypi.org/project/CAMSyFReT/)

# WorldQuant Program Group Projects

[Back to top](#top)

**Timeline: Apr 2025 - Present**

Some of the group work projects I participated in at WorldQuant University involved collaborative code implementations and the preparation of comprehensive final reports as the primary deliverables. These projects focused on applied quantitative finance and machine learning skills. They include, but are not limited to:

- [Building Machine Learning Models from Scratch](#building-machine-learning-models-from-scratch)
- [Multi-Asset Prediction](#multi-asset-prediction)
- [Backtesting and Validation](#backtesting-and-validation)
- [Quantitative Model Survey](#quantitative-model-survey)
- [Stock Market Prediction in Emerging Markets (Paper Implementation)](#stock-market-prediction-in-emerging-markets-paper-implementation)
- [Hyperparameter Tuning and Generalization](#hyperparameter-tuning-and-generalization)
- [Time Series and Feature Engineering Challenges](#time-series-and-feature-engineering-challenges)
- [Regime-Based Allocation Strategy](#regime-based-allocation-strategy)
- [Data Quality and Yield Curve Modeling](#data-quality-and-yield-curve-modeling)
- [Empirical Analysis of ETFs](#empirical-analysis-of-etfs)
- [Option Pricing](#option-pricing)
- [Omitted Variable Bias and Specification](#omitted-variable-bias-and-specification)
- [Time Series Stationarity and Structural Breaks](#time-series-stationarity-and-structural-breaks)

## Building Machine Learning Models from Scratch

Implemented core machine learning algorithms from first principles using only fundamental libraries. Developed Linear Regression, fully connected Neural Networks, Convolutional and Recurrent Neural Networks, ARIMA models, etc to gain deep understanding of their internal mechanics, optimization processes, and mathematical foundations.

**Focus Areas**: machine learning fundamentals, linear regression, neural networks, backpropagation, recurrent neural networks, time series modeling, optimization algorithms, linear algebra, matrix decompositions

**Tech Stack**: `Python` `Numpy` `Matplotlib` `Pandas` `Scikit-learn` `tensorflow` `pytorch`

**Key Learning**: linear regression with gradient descent, loss and cost functions, forward and backward propagation, neural network architecture implementation, RNNs for sequential data, ARIMA model mechanics, manual gradient computation and optimization

## Multi-Asset Prediction

Multi-asset deep learning modeling for tactical asset allocation. Built and compared three DL architectures (MLP, CNN-GAF, LSTM) to predict 25-day ahead returns for five major ETF asset classes (SPY, TLT, SHY, GLD, DBO). Developed a multi-output LSTM to jointly predict returns across assets and capture cross-asset dependencies. Implemented a practical trading strategy based on model predictions and conducted comprehensive backtesting against an equally weighted buy-and-hold benchmark.

**Focus Areas**: multi-asset forecasting, architecture comparison (MLP vs CNN vs LSTM), multi-output models, cross-asset dependencies, portfolio strategy design, backtesting, tactical asset allocation, feature engineering

**Tech Stack**: `Python` `TensorFlow` `Keras` `Pandas` `NumPy` `Scikit-Learn` `yfinance` `Matplotlib` `Seaborn` `Plotly`

**Key Learning**: time series prediction across heterogeneous asset classes, Gramian Angular Field (GAF) image transformations for time series encoding, multi-task learning and multi-output architectures, practical trading strategy implementation, backtesting framework design, limitations of prediction accuracy in achieving portfolio performance

## Backtesting and Validation

Demonstrated the critical importance of proper validation design in financial model backtesting through structured analysis of Bitcoin price prediction. Intentionally introduced temporal leakage in the initial train/test split to establish an inflated baseline, then progressively applied walk-forward validation and purging/embargo techniques to quantify performance degradation. This project isolates the impact of validation methodology from model capability, showing how poor validation design can create misleading backtesting results.

**Focus Areas**: backtesting methodology, temporal data leakage, walk-forward validation, validation design impact, purging and embargo techniques, realistic out-of-sample performance assessment, market regime analysis

**Tech Stack**: `Python` `TensorFlow` `Keras` `Pandas` `NumPy` `Scikit-Learn` `yfinance` `Matplotlib` `Seaborn`

**Key Learning**: temporal data leakage identification and mitigation strategies, walk-forward vs static validation trade-offs, purging (removing label-leaky samples) and embargo (buffering periods) implementation, validation design sensitivity in financial forecasting, market regime exposure across test periods, realistic vs optimistic performance assessment, importance of chronological ordering in time series validation

## Quantitative Model Survey

Created a structured survey of finance-focused ML techniques, covering regularized linear models, clustering, PCA, tree-based methods, discriminant analysis, support vector machines, and neural networks, comparing advantages, limitations, and application scenarios.

**Focus Areas**: regression, clustering, dimensionality reduction, tree models, SVM, neural networks

**Tech Stack**: `Python` `Pandas` `Numpy` `Scikit-Learn` `YFinance` `Seaborn` `Matplotlib`

**Key Learning**: model selection, regularization, comparative evaluation, and financial ML strategy.

## Stock Market Prediction in Emerging Markets (Paper Implementation)

Conducted an in-depth academic review and practical replication of the paper “An Intelligent Approach for Predicting Stock Market Movements in Emerging Markets Using Optimized Technical Indicators and Neural Networks” (Sagaceta Mejia et al.). Implemented optimized indicator-based feature engineering with neural networks to forecast stock price movements, achieving performance consistent with published results.

**Focus Areas**: data understanding, feature selection, cross validation and modeling, paper replication and review

**Tech Stack**: `Python` `Pandas` `Numpy` `Scikit-Learn` `YFinance` `Seaborn` `Matplotlib`

**Key Learning**: paper review

## Hyperparameter Tuning and Generalization

Focused on validation, hyperparameter optimization, and the bias-variance tradeoff to improve out-of-sample performance of machine learning models. The project also explored ways to combine models for more robust results.

**Focus Areas**: model tuning, validation analysis, ensemble reasoning, hyperparameter optimization.

**Tech Stack**: `Python` `Pandas` `Numpy` `Scikit-Learn` `YFinance` `Seaborn` `Matplotlib`

**Key Learning**: bias-variance tradeoff, hyperparameter optimization, and model generalization.

## Time Series and Feature Engineering Challenges

Addressed time series challenges including feature extraction, non-stationarity, equilibrium modeling, multicollinearity, and regime detection. The emphasis was on selecting appropriate models and explaining their financial implications.

**Focus Areas**: time-series econometrics, feature engineering, regime identification

**Tech Stack**: `Python` `Pandas` `Numpy` `Scikit-Learn` `YFinance` `Seaborn` `Matplotlib` `statsmodels` `hmmlearn` `fredapi` `ARIMA`

**Key Learning**: feature selection, handling multicollinearity, regime-change detection, ARIMA modeling.

## Regime-Based Allocation Strategy

Designed a volatility-regime allocation framework with Markov chains and Hidden Markov Models. The project covered data preparation, regime identification, and rule-based allocation logic.

**Focus Areas**: regime switching, Markov chains, HMMs, asset allocation, backtesting and allocation strategy evaluation.

**Tech Stack**: `Python` `Pandas` `Numpy` `Scikit-Learn` `YFinance` `Seaborn` `Matplotlib` `plotly` `hmm models`

**Key Learning**: volatility regime modeling, regime-driven allocation, hidden markov models, time-series strategy design, AIC and BIC criterion, Log-likelihood.

## Data Quality and Yield Curve Modeling

Assessed structured and unstructured financial data quality, then modeled government bond term structures using Nelson-Siegel and cubic spline approaches. The project compared model fit across short- and long-term maturities, evaluated goodness-of-fit, analyzed parameter interpretations, and discussed the ethical implications of smoothing yield curves in financial reporting.

**Focus Areas**: fixed income modeling, term structure analysis, model comparison

**Tech Stack**: `Python` `Pandas` `Numpy` `fredapi` `matplotlib` `seaborn` `nelson_siegel_svensson` `scipy`

**Key Learning**: data quality evaluation, yield curve fitting, model interpretation, parameter estimation, ethical considerations in quantitative finance, and financial data engineering.

## Empirical Analysis of ETFs

Performed an in-depth analysis of Exchange-Traded Funds (ETFs), applying transformations and presenting results through clear, concise reports.

**Focus Areas and Key Learnings**: PCA, SVD, covariance matrix analysis, dimensionality reduction in portfolio context, financial returns calculation, in-depth mathematical interpretation.

**Tech Stack**: `Python` `Pandas` `Numpy` `Scikit-Learn` `YFinance` `Matplotlib`

## Option Pricing

Validated vanilla option prices using binomial and trinomial tree models, applied put-call parity checks, compared European and American option valuations, and computed Greeks for risk assessment. Also assessed option pricing using Heston and Merton frameworks, with Monte Carlo simulation for volatility and jump dynamics. The project analyzed pricing under different correlation regimes and calculated Greeks.

**Focus Areas**: options pricing (American, European, and Asian), tree-based valuation, Black-Scholes, risk sensitivities, stochastic volatility (Heston Modeling), jump-diffusion models (Merton Modeling), Monte Carlo simulation

**Tech Stack**: `Python` `Pandas` `Numpy` `plotly` `Matplotlib` `Scipy`

**Key Learning**: pricing model validation, parity relationships, Greek computation and delta hedging, advanced option modeling, volatility regime analysis, and risk metric computation.

## Omitted Variable Bias and Specification

Analyzed regression misspecification by comparing models with and without omitted variables, using simulation to demonstrate the effects on parameter estimates and inference.

**Focus Areas**: econometric diagnostics, simulation-based analysis, regression modeling

**Tech Stack**: `Python` `Pandas` `Numpy` `Matplotlib` `statsmodels`

**Key Learning**: omitted variable bias, model specification, parameter interpretation, effects of outliers on regression models, forward and backward selection regression, AIC criterion

## Time Series Stationarity and Structural Breaks

Investigated key time series properties and regression stability issues. Analyzed stationarity through unit root testing on real equity data and examined the implications of non-stationary processes. Additionally, developed a dummy variable approach to detect and test for a structural break in a linear regression model.

**Focus Areas**: time series analysis, stationarity, unit root testing, structural breaks, dummy variable regression, econometric diagnostics

**Tech Stack**: `Python` `Pandas` `Numpy` `Matplotlib` `statsmodels` `yfinance`

**Key Learning**: unit root vs. explosive roots and their simulation behavior, Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test, consequences of non-stationarity in finance and econometrics, structural break detection using interaction terms, proper model specification for regime shifts, autocorelation function (ACF) and PACF

# Frontend Engineering

[Back to top](#top)

Some of my frontend projects, they include, but are not limited to:

- [Yivera](#yivera)
- [Cashout-Net - Mar 2025 - Oct 2025](#cashout-net)
- [Syftset v2 - Dec 2024 - Mar 2025](#syftset-v2)
- [Risevest Landing - Oct 2024](#risevest-landing)
- [Simbrella - Jul 2024](#simbrella)
- [Angular Test - Nov 2022](#angular-test)
- [Lendsqr Frontend Test - Nov 2022](#lendsqr-frontend-test)
- [Syftset v1 - Jun 2022 - Nov 2022](#syftset-v1---2022)
- [Ecommerce Application - Mar 2022 - Aug 2022](#ecommerce-application)
- [Unique Help Global Concept - Dec 2020 - Feb 2021](#unique-help-global-concept)

## Yivera

Timeline: Jul 2022 - Present

Worked on frontend/admin build of the Yivera music distribution platform.

**Focus Areas**: Dashboard systems, API integration, UI responsiveness, State management, Performance debugging, System Monitoring, Ongoing Website Management, Monorepo setups, Payment Gateway setups \[Stripe, Paypal, Flutterwave, Crypto, others\], Authentication and Authorization, CI/CD, PWA

**Tech Stack**: `React` `Next.js` `TypeScript` `JavaScript` `SCSS` `Redux` `tanstack` `MUI` `Tailwind` `formik` `yup` `axios` `Google Analytics` `Digital Ocean` `Vercel` `Render` `Figma` `Postman`

**Links**

- Website: [Visit Yivera](https://www.yivera.com/)

## Cashout-Net

Timeline: Mar 2025 - Oct 2025

Worked on the frontend/admin build of the cashout-net platform, an online raffle draw platform, for a nigerian radio station.

**Focus Areas**: Dashboard systems, API integration, UI responsiveness, State management, Performance debugging, System Monitoring, Website Maintenance, Payment Gateway setups \[Opay, Paystack, USSD\], Authentication and Authorization, CI/CD

**Tech Stack**: `Next.js` `TypeScript` `MUI` `react-hook-form` `zod` `Render` `Figma` `Swagger`

## Syftset v2

Timeline: Dec 2024 - Mar 2025

A sleek frontend that retrieves and visualizes key trading metrics from Firestore. Designed for investors to track performance effortlessly.

**Focus Areas**: Dashboard systems, UI responsiveness, Performance debugging, System Monitoring, Website Maintenance, User Authentication, CI/CD, Backend-as-a-Service

**Tech Stack**: `Next.js` `TypeScript` `MUI` `firebase` `vercel` `Google Analytics`

**Links**

- Website: [Visit Syftset](https://www.syftset.com/)
- Dashboard: [View Accessible Dashboard](https://syftsetfrontend.vercel.app)
  - Login Details:
    - Email: crypt.syftset.tech@gmail.com
    - Password: syft_set@98A
- Github: [Dummy Repo - Repository Link](https://github.com/The-Professor99/Syftset_Frontend)
- [Syftset v2 - Dummy Backend](#syftset-v2---dummy-backend---feb-2025)
- [Syftset Core Application](#syftset-core-application)
- [Syftset v1 Frontend](#syftset-v1---2022)
- [Syftset v1 Backend](#syftset-v1---backend---2022)

## Risevest Landing

Timeline: Oct 2024

**Tech Stack**: `Nextjs` `HTML` `SCSS` `Typescript`

**Links**

- Website: [Visit Website](https://risevest-frontend.netlify.app/)
- Github: [Repository Link](https://github.com/The-Professor99/risevest_landing)

## Simbrella

Timeline: Jul 2024

**Tech Stack**: `Nextjs` `headlessui` `zod` `firebase` `vercel`

**Links**

- Website: [Visit Website](https://ihechi-festus-simbrella.vercel.app/)
- Github: [Repository Link](https://github.com/The-Professor99/Simbrella)

## Angular Test

Timeline: Nov 2022

Also features a fake JSON Server deployed to heroku.

**Tech Stack**: `Angular2` `Typescript` `RxJS` `Angular MaterialUI`, `NGRX` `Javascript` `JSON-Server` `FakerJS` `netlify` `heroku`

**Links**

- Website: [Visit Website](https://angular-test99.netlify.app/)
- Github: [Repository Link](https://github.com/The-Professor99/Angular-test)
- Github: [Fake API server](https://github.com/The-Professor99/fake_api_server)

## Lendsqr Frontend Test

Timeline: Nov 2022

**Tech Stack**: `React` `Typescript` `Redux` `MaterialUI`

**Links**

- Website: [Visit Website](https://ihechi-festus-lendsqr-fe-test.netlify.app/)
  - Login Details:
    - Email: lendsqr-fe-test@mail.com (or any other email)
    - Password: lend1@98A (or any other password)
- Github: [Repository Link](https://github.com/The-Professor99/lendsqr-fe-test)

## Syftset v1 - 2022

Timeline: Jun 2022 - Nov 2022

Developed the client application for the Syftset platform, providing users with real-time access to prediction signals generated by the [backend service](#syftset-v1---backend---2022). Implemented a Twitter-style social feed that enables users to create and interact with posts within the application. Leveraged Socket.IO to establish WebSocket connections for instant delivery of prediction signals and other real-time updates. Integrated Firebase Authentication for secure user management, Firestore for data persistence, Cloud Messaging for push notifications, and Analytics for user behavior tracking and application performance monitoring.

**Focus Areas**: System Monitoring, Website Maintenance, User Authentication, CI/CD, Backend-as-a-Service, PWA, Push app notifications

**Tech Stack**: `React` `JavaScript` `mui` `scss` `babel` `webpack` `redux` `axios` `formik` `yup` `cypress` `react testing library` `plotly` `react-fetch-hook` `websocket` `socket.io` `firebase` `netlify` `Google Analytics`

**Links**

- Website: [Visit Syftset v1](https://syftset1.web.app/)
- [Syftset Core Application](#syftset-core-application)
- [Syftset v1 Backend](#syftset-v1---backend---2022)

## Ecommerce Application

Timeline: Mar 2022 - Aug 2022

**Focus Areas**: Dashboard systems, API Integration, UI responsiveness, Performance debugging, User Authentication, Payment Gateway setups \[Flutterwave, Stripe\]

**Tech Stack**: `React` `Stripe` `React testing library` `bootstrap` `flutterwave` `formik` `styledcomponents` `webpack`

**Links**

- Website: [Visit Website](https://ecommerce-practice-99.netlify.app/)
  - Login Details
    - Email: sneakers.crypt.ihechi@gmail.com
    - Password: sneak1@98A
  - Account creation is also enabled
- Github: [Repository Link](https://github.com/The-Professor99/ecommerce_practise2)

## Unique Help Global Concept

Timeline: Dec 2020 - Feb 2021

Worked on the frontend/admin build of the unique help platform.

**Focus Areas**: Dashboard systems, API Integration, UI responsiveness, Performance debugging, System Monitoring, Website Maintenance, User Authentication, CI/CD

**Tech Stack**: `Angular2` `typescript` `rxjs` `tailwind` `chartjs` `angular-datatables` `jasmine` `karma` `jquery` `Google Analytics` `Postman` `Figma`

**Links**

- Website: [Visit UHGC's website](https://www.uniquehelp247.com/)

---

# Backend Engineering

[Back to top](#top)

- [Syftset v1 - 2022](#syftset-v1---backend---2022)
- [Ecommerce Application - 2020](#ecommerce-application---2020---2021)

## Syftset v1 - Backend - 2022

Timeline: Jun 2022 - Nov 2022

Built a Flask-SocketIO-powered backend application that serves prediction signals generated by the [Syftset Core Application](#syftset-core-application). Implemented real-time server-client communication to deliver signals instantly, with automated periodic signal generation orchestrated through Heroku Scheduler.

**Focus Areas**: automation, scripting

**Tech Stack**: `Python` `Flask` `Postgresql` `APScheduler` `alembic` `Gunicorn` `websockets` `socket.io` `flask_socketio` `flask_cors` `flask_sqlalchemy` `sqlite3` `Heroku` `Backtesting` `ta` `numpy` `pandas` `requests` `beautifulsoup4`

**Links**

- [Syftset Core Application](#syftset-core-application)
- [Syftset v1 Frontend](#syftset-v1---2022)

## Ecommerce Application - 2020 - 2021

Developed a full-stack e-commerce platform for a clothing retail service as a group project for the EEE 562: Web Based Design and Applications course.

**Focus Areas**: web-based system design, e-commerce development, frontend and backend integration, database management

**Tech Stack**: `Python` `Django`

**Key Learning**: session and cart management, CRUD operations, group project collaboration

---

# Quantitative Finance and Trading Systems

[Back to top](#top)

## Syftset Core Application

Timeline: 2021 - Present

Development of systems and tooling related to algorithmic trading operations and portfolio management. Experimentation with rule-based and ML-assisted forex and crypto trading strategies.

**Focus Areas**: Trading strategy development, Trading automation, Backtesting, Signal generation, Portfolio analytics, Risk management, Performance tracking, Data Analysis, Binance APIs, Bybit APIs, MetaTrader APIs, Machine Learning

**Tech Stack**: `Python` `pandas` `numpy` `scipy` `matplotlib` `backtesting` `ta` `mplfinance` `APIs` `requests` `beautifulsoup4` `sklearn` `scipy` `MetaTrader5` `GCP`

**Links**

- Internal Project / Private
- Backend:
  - [View v1](#syftset-v1---backend---2022)
- Frontend:
  - [View v1](#syftset-v1---2022)
  - [View v2](#syftset-v2)

---

# Python Projects

[Back to top](#top)

- [Syftset v2 - Dummy Backend - Feb 2025](#syftset-v2---dummy-backend---feb-2025)
- [Music Downloader - Dec 2020](#music-downloader---dec-2020)

## Syftset v2 - Dummy Backend - Feb 2025

Designed and implemented a simple investment management "backend" that manages account operations, profit-sharing, referral rewards, trading fees, and session performance tracking. It supports multiple investment account types, maintains detailed transaction histories, and leverages Firebase Authentication and Firestore for secure user management and real-time data persistence and access.

**Focus Areas**: fintech, backend architecture, financial automation, referral systems, transaction processing, database design

**Tech Stack**: `python` `firebase-admin-sdk` `cloud-firestore`

**Links**

- Github: [Repository Link](https://github.com/The-Professor99/Syftset_Dummy_Backend)
- Syftset v2: [View Frontend](#syftset-v2)

## Music Downloader - Dec 2020

A music scraper that utilizes web data scraping techniques to automate the seamless download of songs from music hosting sites, circumventing the need to navigate through ads. Built with Python, leveraging the Requests, Selenium and BeautifulSoup libraries. This tool was conceived and implemented before the widespread adoption of music streaming platforms.

**Focus Areas**: web scrapping, automation, scripting, productivity tooling

**Tech Stack**: `python` `selenium` `beautifulsoup4` `webdriver_manager` `requests`

**Links**

- Github: [Repository Link](https://github.com/The-Professor99/Music_Downloader)

---

# Other Projects

[Back to top](#top)

## Single-Axis Solar Tracker System

Timeline: Aug 2019

Developed a single-axis solar tracker as a school engineering project. The system automatically adjusts the orientation of a solar panel in a horizontal direction to maximize energy capture by continuously tracking the sun’s position throughout the day.

**Focus Areas**: embedded systems, renewable energy, sensor integration, control systems, real-time tracking algorithms, hardware-software interfacing

**Tech Stack**: `Electronic Design Automation (EDA) Softwares` `LDR` `Sensors` `Servo Motors` `DC Motors` `Solar Panel` `MATLAB`

Key Learning: light-dependent resistor (LDR) based sun tracking, power optimization in solar systems, analog sensor calibration, mechanical-electronic system integration, data logging and performance analysis

---

# Courses and Certifications

- [Machine Learning Certifications](#machine-learning-certifications)
- [Frontend Engineering Certifications](#frontend-engineering-certifications)
- [Others](#other-certifications)

## Machine Learning Certifications

- [AI Agents Intensive Vibe Coding Course with Google (Kaggle 5-Day Program) - Jun 2026](#ai-agents-intensive-vibe-coding-course-with-google-kaggle-5-day-program---jun-2026)
- [AI Agents Intensive Course with Google (Kaggle 5-Day Program) - May 2026](#ai-agents-intensive-course-with-google-kaggle-5-day-program---may-2026)
- [Machine Learning with Python - May 2024](#machine-learning-with-python---may-2024)
- [Time Series Forecasting with Machine Learning - Sept 2023](#time-series-forecasting-with-machine-learning---sept-2023)
- [AWS Machine Learning Foundations - Nov 2022](#aws-machine-learning-foundations---nov-2022)

### AI Agents Intensive Vibe Coding Course with Google (Kaggle 5-Day Program) - Jun 2026

Completed Google's 5-day continuation intensive focused on vibe coding workflows and production-grade agent engineering. The program covered moving from prompt-driven prototypes to robust agent systems with tool interoperability, skills, security evaluations, and cloud deployment patterns.

**Focus Areas**: vibe coding, agent interoperability, MCP integration, agent skills and memory, human-in-the-loop workflows, security and evaluation, spec-driven development, production deployment

**Tech Stack**: `Python` `Google ADK (Agent Development Kit)` `Agents CLI` `Google Antigravity` `MCP` `Google AI Studio` `Cloud Run` `Google Cloud`

**Key Learning**: intent-first SDLC patterns, tool and protocol integration (MCP/A2A/A2UI), reusable skills-based agents, secure/tested agent workflows, and spec-driven productionization with observability.

**Projects and Capstone**: explored hands-on codelabs including AI Studio to Cloud Run deployment, MCP server integration, ADK/Agents CLI skill-based agent builds, expense-approval agent with human-in-the-loop triage and local evals, plus optional cloud-hosted frontend + async architecture.

- Type: Course
- Learning Platform: Google (in collaboration with Kaggle)
- Date: Jun 2026
- [Show Learning Badge](https://developers.google.com/profile/badges/events/cloud/five-day-ai-agents)

### AI Agents Intensive Course with Google (Kaggle 5-Day Program) - May 2026

Completed Google's intensive hands-on course on building AI agents. Covered the full lifecycle of designing, developing, and deploying intelligent agents capable of reasoning, planning, using tools, maintaining memory, and collaborating to solve real-world tasks.

**Focus Areas**: AI agents, agentic workflows, tool integration, context engineering, memory management, multi-agent systems, agent evaluation and deployment

**Tech Stack**: `Python` `Google ADK (Agent Development Kit)` `LangChain` `Gemini models` `Google AI studio`

**Key Learning**: core concepts of AI agents, reasoning and planning loops, tool use and interoperability, session & memory management, different agent workflow patterns, building production-ready agents, evaluation techniques for agent performance

- Type: Course
- Learning Platform: Google (in collaboration with Kaggle)
- Date: May 2026

### Machine Learning with Python - May 2024

**Main Focus**: Machine Learning with Python, supervised and unsupervised learning, neural networks, TensorFlow, Keras, image classification, natural language processing, data preprocessing, model evaluation, regression and classification algorithms, building and training deep learning models

- Type: Certification
- Issued By: FreeCodeCamp
- Issue Date: May 2024
- Credential ID: the-professor99-mlwp
- [Show Credential](https://www.freecodecamp.org/certification/The-Professor99/machine-learning-with-python-v7)

### Time Series Forecasting with Machine Learning - Sept 2023

**Main Focus**: linear regression with time series, modeling trend with moving averages, capturing seasonality and trend using indicators and Fourier features, using lag features, hybrid models, forecasting strategies with machine learning, ARIMA modeling, time series analysis and forecasting, feature engineering

- Type: Certification
- Issued By: Kaggle
- Issue Date: Sept 2023
- [Show Credential](https://www.kaggle.com/learn/certification/ihechifestus/time-series)

### AWS Machine Learning Foundations - Nov 2022

**Main Focus**: Machine Learning with AWS, AWS SageMaker, AWS DeepComposer, Generative Adversarial Networks (GANs), Generator and Discriminator architecture, ML techniques and Generative AI, pre-trained deep learning models, Software Engineering best practices, Test-driven development, object-oriented programming (OOP), Python programming

- Type: Certification
- Issued By: Udacity
- Issue Date: Nov 2022
- Credential ID: LCHKAJPT
- [Show Credential](https://www.udacity.com/certificate/LCHKAJPT)

## Frontend Engineering Certifications

### Google Africa Developer Scholarship (GADS) - Mobile Web

**Main Focus**: Mobile Web Development, Responsive Web Design, HTML, CSS, JavaScript, Angular, Progressive Web Apps (PWAs), Performance Optimization, Mobile-first development, Modern web technologies, Building fast and accessible web applications

- Type: Scholarship Program
- Issued By: Google Africa & Pluralsight
- Issue Date: 2020
- [View Credentials](https://www.linkedin.com/in/ihechi-festus/details/certifications/)

## Other Certifications

### Foundations of Financial Engineering

**Main Focus**: Financial Engineering, Computational Finance, Financial Data, Financial Markets, Numerical Linear Algebra, Financial Analytics, Options, Yield Curve Analysis, Financial Products, Financial Data Management

- Type: Certificate
- Issued By: WorldQuant University
- Issue Date: Jul 2025
- [View Credentials](https://www.credly.com/badges/447e1fe4-c467-483b-9ce5-0bd3f89de705)

### Google IT Support Professional Certificate

**Main Focus**: IT fundamentals, Technical support, Computer networking, Operating systems, System administration, IT security, Troubleshooting, Customer service, Python scripting for IT automation

- Type: Certificate
- Issued By: Google (via Coursera)
- Issue Date: 2020
- [View Credentials](https://www.linkedin.com/in/ihechi-festus/details/certifications/)

---

# Education

## WorldQuant University (USA - Online)

- Master's degree, Financial Engineering
- Duration: Apr 2025 – Apr 2027
- Financial Engineering, Machine Learning, Deep Learning, Portfolio Management, Risk Management, etc.

## University of Jos, Nigeria

- Bachelor of Engineering - BE, Electrical and Electronics Engineering
- Duration: 2015 – 2021
- Grade: 4.18/5.00
- Research Skills and Engineering, Artificial Neural Networks and Fuzzy Systems, etc.

---

# Hands on Learning

[Back to top](#top)

A set of tutorials and notebooks exploring agent orchestration, transformer workflows with LangChain, LangGraph, Haystack, and Hugging Face tools, machine learning and deep learning techniques.

- [Advanced Multi-Agent Workflow with LangGraph - May 2026](#advanced-multi-agent-workflow-with-langgraph---may-2026)
- [Agent Retrieval and Tool Integration - Oct 2024](#agent-retrieval-and-tool-integration---oct-2024)
- [LangChain Chatbot Prototyping - Sept 2024](#langchain-chatbot-prototyping---sept-2024)
- [Text Clustering and Topic Discovery - Aug 2024](#text-clustering-and-topic-discovery---aug-2024)
- [Conversational Memory and Agent Refresher - Aug 2024](#conversational-memory-and-agent-refresher---aug-2024)
- [Text Summarization Workflow - Aug 2024](#text-summarization-workflow---aug-2024)
- [LangChain Fundamentals and Deployment Concepts - Dec 2023](#langchain-fundamentals-and-deployment-concepts---dec-2023)
- [Hands-On ML with Scikit-Learn - 2022/2023](#hands-on-ml-with-scikit-learn---2022-2023)

---

## Advanced Multi-Agent Workflow with LangGraph - May 2026

Explores a stateful multi-agent architecture built with LangGraph, including a supervisor/orchestrator pattern, specialized subagents, human-in-the-loop debugging, and structured performance evaluation. The notebook focuses on reducing hallucinations, managing conversation flow, and instrumenting the agent pipeline with LangSmith-style observability.

**Tech Stack**: `LangGraph`, `LangChain`, `LangSmith`, `OpenAI-compatible chat models` `sqlite3` `sqlalchemy`

**Key Learning**: hierarchical agent design, stateful workflows, multi-agent coordination, and debugging/metrics for agentic applications.

**Links**

- Colab: [View Notebook](https://colab.research.google.com/drive/1txUROkx_Tv9baBcVFRLYRhqlw4xrb5N2?usp=sharing)

## Agent Retrieval and Tool Integration - Oct 2024

Demonstrates how to build retrieval-enabled agents, convert retrievers into tools, and let agents manage retrieval decisions. Includes examples of when an agent does or does not execute retrieval based on query intent.

**Tech Stack**: `LangChain`, `LangSmith`, `LLMs`

**Key Learning**: tool-enabled agents, conversational retrieval, and retrieval-augmented generation behavior.

**Links**

- Github: [View Notebook](/hands_on_learning/transformers_agents_multiagents/Langchain_Learn.ipynb)

## LangChain Chatbot Prototyping - Sept 2024

Hands-on chatbot experiments that demonstrate prompt design, model definition, and agent-based conversation flow. These notebooks serve as practical prototypes for building conversational assistants with LangChain.

**Tech Stack**: `LangGraph`, `LLMs`

**Key Learning**: agent-driven chatbots, prompt engineering, conversational tool usage.

**Links**

- Github: [View Notebook 1](/hands_on_learning/transformers_agents_multiagents/Chatbot.ipynb)
- Github: [View Notebook 2](/hands_on_learning/transformers_agents_multiagents/Chatbot_Copy1.ipynb)
- Github: [View Notebook 3](/hands_on_learning/transformers_agents_multiagents/Chatbot_2.ipynb)

## Text Clustering and Topic Discovery - Aug 2024

Implements document clustering using Bag-of-Words, TF-IDF/Hashing feature extraction, LSA dimensionality reduction, and KMeans clustering. Includes cluster evaluation with homogeneity, completeness, V-measure, Adjusted Rand Index, and silhouette analysis.

**Tech Stack**: `scikit-learn`, `KMeans`, `TF-IDF`, `HashingVectorizer`, `LSA`

**Key Learning**: text clustering pipelines, sparse data clustering, and quality metrics for unsupervised NLP.

**Links**

- Github: [View Notebook](/hands_on_learning/transformers_agents_multiagents/KMeans_Text_Clustering.ipynb)

## Conversational Memory and Agent Refresher - Aug 2024

Reviews message history, stateful conversation handling, session IDs, how to wrap language models with memory for chat applications, and agents.

**Tech Stack**: `LangChain`, `LangSmith`, `LLMs`

**Key Learning**: session history, stateful chat workflows, and memory-backed agent patterns.

**Links**

- Github: [View Notebook](/hands_on_learning/transformers_agents_multiagents/Langchain_Refresher.ipynb)

## Text Summarization Workflow - Aug 2024

Explores summarization of long articles with transformer models, handling context limits, data preparation, and evaluation heuristics such as newline-separated summaries and sentence boundary handling.

**Tech Stack**: `Huggingface datasets` `Huggingface transformers` `nltk`

**Key Learning**: summarization dataset prep, long-context limitations, and baseline model understanding.

**Links**

- Colab: [View Notebook](https://colab.research.google.com/drive/1ZS0KoNgq9sWdoaSoC2azK4SfuPulhvfW?usp=sharing)

## LangChain Fundamentals and Deployment Concepts - Dec 2023

Covers core LangChain concepts such as chains, prompts, output parsers, ChatModel vs LLM, LangSmith debugging, and LangServe deployment. Also compares open-source model usage with managed OpenAI workflows.

**Tech Stack**: `LangChain`, `LangSmith`, `LangServe` `FastAPI` `OpenAI-compatible chat models`

**Key Learning**: chain construction, prompt templates, message roles, and LangChain deployment patterns.

**Links**

- Github: [View Notebook](/hands_on_learning/transformers_agents_multiagents/Langchain.ipynb)
- Github: [View Backend Example](/hands_on_learning/transformers_agents_multiagents/langserve_with_user_id_chat_persistence.py)

## Hands-On ML with Scikit-Learn - 2022-2023

A collection of tutorials based on Aurelien Geron's Hands-On Machine Learning with Scikit-Learn. These notebooks follow the book's chapters and provide practical exercises for supervised learning, model evaluation, ensemble methods, and end-to-end ML pipelines.

**Scope**: supervised learning, unsupervised learning, validation, feature engineering, ensemble methods, neural network, deep learning, and reinforcement learning workflows.

**Key Learning**: machine learning best practices, applied book-based tutorials, model validation strategies, and end-to-end machine learning development.

**Links**

- Github: [View Notebooks](/hands_on_learning/hands_on_ml_with_scikit-learn_Aurelien_textbook)

---

# Contact

- Website: [Visit My Portfolio website](https://ihechifestus9.web.app)
- GitHub: [Visit My GitHub Profile](https://github.com/The-Professor99)
- LinkedIn: [Connect on LinkedIn](https://www.linkedin.com/in/ihechi-festus/)
- X/Twitter: [Connect on X](https://x.com/FestusIhechi)
- Email: [Send me an Email](ihechifestus999@gmail.com)

---

# Notes

Most repositories here are focused on:

- learning by building,
- research reproduction,
- practical experimentation,
- and real-world engineering applications.

**_Some projects are experimental or ongoing and may continue evolving over time._**

<a id="bottom"></a>
[Back to top](#top)
