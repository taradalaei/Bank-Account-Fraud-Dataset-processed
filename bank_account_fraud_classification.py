import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from lazypredict.Supervised import LazyClassifier


df = pd.read_csv("D:/uni_st/term 6/AI/hws/hw5/cleaned_preprocessed_dataset.csv")


# Data normalization
# Min-Max scaling (normalize between 0 and 1) for numerical features
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

#split the labels from our dataset
target = 'fraud_bool'
X = df.drop(target,axis=1)
y = df[target]
print(X.head())

# Train/test split with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
train = pd.concat([X_train, y_train], axis=1).copy()
train_copy = pd.concat([X_train, y_train], axis=1).copy()

# Initialize LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

# Fit and predict using LazyClassifier
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# Display the results
print(models)

# Initialize the classifier
knn = KNeighborsClassifier()

# Fit the model
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Calculate and print the metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Cross-validation
cv_scores = cross_val_score(knn, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {np.mean(cv_scores)}")

# Apply PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Initialize the classifier
knn = KNeighborsClassifier()

# Fit the model
knn.fit(X_train_pca, y_train)

# Predict on the test set
y_pred_pca = knn.predict(X_test_pca)

# Calculate and print the metrics
accuracy_pca = accuracy_score(y_test, y_pred_pca)
f1_pca = f1_score(y_test, y_pred_pca)
precision_pca = precision_score(y_test, y_pred_pca)
recall_pca = recall_score(y_test, y_pred_pca)
conf_matrix_pca = confusion_matrix(y_test, y_pred_pca)

print(f"Accuracy (PCA): {accuracy_pca}")
print(f"F1 Score (PCA): {f1_pca}")
print(f"Precision (PCA): {precision_pca}")
print(f"Recall (PCA): {recall_pca}")
print(f"Confusion Matrix (PCA):\n{conf_matrix_pca}")

# Cross-validation
cv_scores_pca = cross_val_score(knn, pca.fit_transform(X), y, cv=5)
print(f"Cross-validation scores (PCA): {cv_scores_pca}")
print(f"Mean cross-validation score (PCA): {np.mean(cv_scores_pca)}")
