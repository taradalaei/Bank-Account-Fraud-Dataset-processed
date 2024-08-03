# Bank Account Fraud Detection

This repository contains a comprehensive analysis of the Bank Account Fraud Dataset Suite. The project involves data preprocessing, exploratory data analysis (EDA), clustering, and classification to identify fraudulent transactions. The code demonstrates various data cleaning techniques, dimensionality reduction, and machine learning algorithms for classification.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
5. [Clustering](#clustering)
6. [Classification](#classification)
7. [Results](#results)
8. [License](#license)

## Prerequisites

Ensure you have the following Python packages installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`
- `lazypredict`

You can install these packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy lazypredict
```

## Dataset

The dataset used for this project is the **Bank Account Fraud Dataset Suite** provided for NeurIPS 2022. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022). Ensure the dataset is placed at the right path in your local environment.

## Data Preprocessing

1. **Loading the Dataset**: The dataset is loaded into a Pandas DataFrame.
2. **Handling Missing Values**: Missing values are imputed using the mode of the respective columns.
3. **Outlier Detection**: Outliers are identified using the Interquartile Range (IQR) and are replaced with NaN.
4. **Normalization**: Numerical features are normalized using Min-Max scaling.
5. **Categorical Encoding**: Categorical variables are encoded using one-hot encoding.

## Exploratory Data Analysis (EDA)

1. **Distribution of Target Variable**: The distribution of fraudulent vs. non-fraudulent transactions is visualized.
2. **Correlation Matrix**: A correlation matrix for numerical columns is computed and visualized.
3. **Normality Check**: Q-Q plots and statistics for selected features are generated to check normality.

## Clustering

1. **Standardization**: Features are standardized using `StandardScaler`.
2. **PCA**: Principal Component Analysis (PCA) is applied to retain 95% of variance.
3. **K-Means Clustering**: K-Means clustering is performed with 5 clusters. The clusters are analyzed and visualized using scatter plots.

## Classification

1. **Model Training**: A variety of classifiers, including K-Nearest Neighbors (KNN), are trained on the dataset.
2. **Performance Metrics**: The models are evaluated using accuracy, F1 score, precision, recall, and confusion matrix.
3. **Cross-Validation**: Cross-validation is performed to assess the model's performance.
4. **PCA with Classification**: PCA is applied before classification to evaluate its impact on model performance.

## Results

The results include visualizations and performance metrics for clustering and classification models. These metrics help evaluate the effectiveness of the models and provide insights into the data.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.