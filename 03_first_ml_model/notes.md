# Lesson 03: Selecting Data for Modeling

## Overview
When datasets have too many variables, it's hard to understand them all. In this lesson, you'll learn how to:  
1. Explore the columns in a dataset  
2. Select your **prediction target** (y)  
3. Select **features** (X)  
4. Fit a simple Decision Tree model and make predictions  

We'll use the Melbourne housing dataset as an example.

---

## Step 1: Load the dataset

```python
import pandas as pd
```

# Load the Melbourne housing dataset
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# View all column names
```python
melbourne_data.columns
```

# Explanation:
pd.read_csv() reads the CSV into a Pandas DataFrame.
.columns shows all column names so you can select which variables to use.

## Step 2: Handle missing values
```python
# Drop rows with missing values for simplicity
melbourne_data = melbourne_data.dropna(axis=0)
```

# Explanation:
.dropna(axis=0) removes any rows with missing values.
This is a simple approach for now; later, you can learn more sophisticated imputation techniques.


