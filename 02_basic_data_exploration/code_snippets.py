# Lesson 2: Basic Data Exploration

import pandas as pd

# Load Melbourne dataset
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.head())

# Load Ames Housing dataset
train_file = '../input/ames-housing/train.csv'
train_data = pd.read_csv(train_file)
print(train_data.head())

# Summary statistics
print(melbourne_data.describe())
print(train_data.describe())

# Check missing values
print(melbourne_data.isnull().sum())
print(train_data.isnull().sum())

# Column types
print(melbourne_data.dtypes)
print(train_data.dtypes)
