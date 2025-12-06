# MIS3_assignment ("Hybrid Machine Learning Model for Stock Trend Prediction")

This project implements a machine learning pipeline to predict **stock price movement (up/down)** using:

- 🪟 Sliding Window Dataset Generation  
- ⚡ Fast Fourier Transform (FFT)  
- 🔻 Truncated SVD for Dimensionality Reduction  
- 🤖 SVM (Support Vector Machine) Classification  

The goal is to transform stock closing prices into frequency-domain features and classify whether the price will rise on the next day.

---

## 📌 Features

✔ Converts time-series stock data into supervised learning format  
✔ Uses **FFT** to capture frequency-based patterns  
✔ Compresses feature space using **SVD**  
✔ Classifies movement using **RBF-SVM**  
✔ Achieves competitive accuracy depending on dataset  

---

## 📂 Project Structure
│── Stock_Prices.csv # Input dataset with 'Close' column
│── MIS3_Assignment.py # Full machine learning pipeline
│── README.md # Documentation (this file)

## Invidual Contributions
| Name                     | ID               | Contributions                                      |
| ------------------------ | ---------------- | -------------------------------------------------- |
| **P. Bhargav Ram**       | DL.AI.U4AID24146 | Sliding window, FFT integration                    |
| **Vishnu Karthik C**     | DL.AI.U4AID24109 | SVD, metrics analysis                              |
| **Pujya Ganesh Karthik** | DL.AI.U4AID24141 | SVM classifier, preprocessing                      |
| **Amith Vignesh K**      | DL.AI.U4AID24116 | FFT features, visualizations, pipeline structuring |
