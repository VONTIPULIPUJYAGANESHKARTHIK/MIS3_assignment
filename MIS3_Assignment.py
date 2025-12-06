import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# 1. Data Loading and Preprocessing (Sliding Window)
def create_dataset(data, window_size=30):
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        window = data.iloc[i : i + window_size]['Close'].values
        # Target: 1 if next Close > current Close
        target = 1 if data.iloc[i + window_size + 1]['Close'] > \
                      data.iloc[i + window_size]['Close'] else 0
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)


# Load Data
df = pd.read_csv('Stock_Prices.csv')
X_raw, y = create_dataset(df)


# Sequential Split (80% train, 20% test)
split_idx = int(0.8 * len(X_raw))
X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]


# 2. Feature Extraction (FFT)
def apply_fft(X):
    # Apply FFT to each 30-day window
    fft_features = np.fft.fft(X, axis=1)
    # Return magnitude of the first half (symmetry property)
    return np.abs(fft_features[:, 1:16])

X_train_fft = apply_fft(X_train_raw)
X_test_fft = apply_fft(X_test_raw)


# 3. Dimensionality Reduction (SVD)
n_components = 10
svd = TruncatedSVD(n_components=n_components, random_state=42)
X_train_svd = svd.fit_transform(X_train_fft)
X_test_svd = svd.transform(X_test_fft)


# 4. Classification (SVM)
svm_model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
svm_model.fit(X_train_svd, y_train)


# 5. Results
y_pred = svm_model.predict(X_test_svd)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
