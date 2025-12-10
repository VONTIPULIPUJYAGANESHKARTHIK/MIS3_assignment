# MIS3_assignment ("Hybrid Machine Learning Model for Stock Trend Prediction")

## Introduction
Predicting financial markets is a challenging task due to their extreme volatility and constant fluctuations. When we train directly on raw price data, traditional machine-learning models frequently fail because they overfit. By creating a hybrid prediction pipeline that integrates mathematical tools from Unit 2, this project aims to address this problem. That is, first, we move the data from the time domain to the frequency domain using the \textbf{Fast Fourier Transform (FFT)}. Next, we reduce the dimensionality of the data by using \textbf{Singular Value Decomposition (SVD)}, retaining only the useful data, and finally, we use a \textbf{Support Vector Machine (SVM)} to determine whether the stock will trend "Up" or "Down" the following day.

Overall, our hypothesis is that the classifier will perform better than models trained directly on noisy price data, because cleaning and refining the data first should give it a much clearer signal to learn from. 

The goal is to transform stock closing prices into frequency-domain features and classify whether the price will rise on the next day.

## Invidual Contributions
| Name                     | ID               | Contributions                                      |
| ------------------------ | ---------------- | -------------------------------------------------- |
| **P. Bhargav Ram**       | DL.AI.U4AID24146 | Sliding window, FFT integration                    |
| **Vishnu Karthik C**     | DL.AI.U4AID24109 | SVD, metrics analysis                              |
| **Pujya Ganesh Karthik** | DL.AI.U4AID24141 | SVM classifier, preprocessing                      |
| **Amith Vignesh K**      | DL.AI.U4AID24116 | FFT features, visualizations, pipeline structuring |
