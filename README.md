# MIS3_assignment ("Hybrid Machine Learning Model for Stock Trend Prediction")

## Introduction

The volatile and non-stationary nature of financial time-series data often leads to poor performance and overfitting in traditional machine learning models. The primary goal of this project is to develop a sophisticated, hybrid predictive framework that leverages advanced mathematical techniques to mitigate the effects of market noise and extract meaningful, periodic signals, thereby improving the prediction of the following day's stock market trend (either "up" or "down").

We design a three-stage **Hybrid Pipeline** combining concepts from Signal Processing (Unit 2) and Linear Algebra (Unit 1) with Machine Learning:

1. **Feature Extraction:** The **Discrete Fourier Transform (DFT)** is used to convert the time-domain price history into the frequency domain, effectively isolating underlying periodic market behaviors (low-frequency components) from random daily fluctuations (high-frequency noise).

2. **Dimensionality Reduction:** The high-dimensional frequency features are compressed using **Singular Value Decomposition (SVD)** to obtain the optimal low-rank representation.

3. **Classification:** The reduced features are then used to classify the next day's movement using a **Support Vector Machine (SVM)** classifier.

Our hypothesis is that integrating these mathematical techniques will enhance the stability and generalization capabilities of the predictive model compared to models trained on raw price data.
 

The goal is to transform stock closing prices into frequency-domain features and classify whether the price will rise on the next day.

## Invidual Contributions
| Name                     | ID               | Contributions                                      |
| ------------------------ | ---------------- | -------------------------------------------------- |
| **P. Bhargav Ram**       | DL.AI.U4AID24146 | Sliding window, FFT integration                    |
| **Vishnu Karthik C**     | DL.AI.U4AID24109 | SVD, metrics analysis                              |
| **Pujya Ganesh Karthik** | DL.AI.U4AID24141 | SVM classifier, preprocessing                      |
| **Amith Vignesh K**      | DL.AI.U4AID24116 | FFT features, visualizations, pipeline structuring |
