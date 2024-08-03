import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import scipy.stats as stats

def check_normality(features, data_frame):
    for feature in features:
        plt.figure(figsize = (8,8))
        ax1 = plt.subplot(1,1,1)
        stats.probplot(data_frame[feature], dist=stats.norm, plot=ax1)
        ax1.set_title(f'{feature} Q-Q plot', fontsize=20)
        sns.despine()

        mean = data_frame[feature].mean()
        std = data_frame[feature].std()
        skew = data_frame[feature].skew()
        print(f'{feature} : mean: {mean:.4f}, std: {std:.4f}, skew: {skew:.4f}')
        plt.show()

# Load the dataset
df = pd.read_csv("D:/uni_st/term 6/AI/hws/hw5/bank_account_fraud_code/Bank Account Fraud Dataset Suite (NeurIPS 2022)/Base.csv")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Get information about the dataset
print("\nInformation about the dataset:")
print(df.info())

# Summary statistics of numerical columns
print("\nSummary statistics of numerical columns:")
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

df.drop('device_fraud_count', axis=1, inplace=True)
df.drop('velocity_6h', axis=1, inplace=True)
df.drop('bank_branch_count_8w', axis=1, inplace=True)


# Visualize the distribution of the target variable ('fraud_bool')
plt.figure(figsize=(8, 6))
df['fraud_bool'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Distribution of Fraudulent Transactions')
plt.xlabel('Fraudulent')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Calculate and visualize correlation matrix for numerical columns only
numerical_df = df.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
corr = numerical_df.corr()

plt.figure(figsize=(10, 8))
plt.matshow(corr, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.colorbar()
plt.show()

# Check for duplicate rows
duplicate_rows = df[df.duplicated()]

# Display duplicate rows (if any)
if not duplicate_rows.empty:
    print("Duplicate rows:")
    print(duplicate_rows)
else:
    print("No duplicate rows found.")

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Reset index after dropping duplicates
df.reset_index(drop=True, inplace=True)

# Replace -1 with NaN for features where -1 represents null values
features_with_null = [
    'prev_address_months_count', 
    'current_address_months_count', 
    'intended_balcon_amount', 
    'bank_months_count',
    'session_length_in_minutes'
]
df[features_with_null] = df[features_with_null].replace(-1, np.nan)

#true missing values
print("\nTrue Missing values:")
print(df.isnull().sum())

# Impute missing values
# Using mean imputation for demonstration purposes
df.fillna(df.mode().iloc[0], inplace=True)
print("fill the missing values")

#true missing values after mean inputation
print("\nTrue Missing values after mean imputation:")
print(df.isnull().sum())

# Select numerical columns for outlier detection and handling
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Calculate the IQR for each numerical column
Q1 = df[numerical_cols].quantile(0.25)
Q3 = df[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
print("IQR: ", IQR)

# Define a threshold to identify outliers (e.g., 1.5 times the IQR)
threshold = 1.5

# Identify outliers
outliers = (df[numerical_cols] < (Q1 - threshold * IQR)) | (df[numerical_cols] > (Q3 + threshold * IQR))
print("outliers:", outliers)

# Replace outliers with NaN or remove them
df[numerical_cols][outliers] = np.nan

#fill the nan values again
df.fillna(df.mode().iloc[0], inplace=True)

# Encode categorical variables if any
# Using one-hot encoding for demonstration purposes
df = pd.get_dummies(df, columns=['payment_type', 'employment_status', 'housing_status', 'source', 'device_os'], drop_first=True)


#check normality
features = ['days_since_request', 'zip_count_4w', 'proposed_credit_limit']
check_normality(features, df)

# Display the cleaned and preprocessed dataset
print("Cleaned and Preprocessed Dataset:")
print(df.head())

# Save the cleaned and preprocessed dataset to a new CSV file
df.to_csv('D:/uni_st/term 6/AI/hws/hw5/cleaned_preprocessed_dataset.csv', index=False)
print(df.head)
#####################################################################################
