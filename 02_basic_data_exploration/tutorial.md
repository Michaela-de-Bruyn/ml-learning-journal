# Lesson 2 Tutorial: Basic Data Exploration

This tutorial guides you through exploring datasets using Python's Pandas library. Each code snippet is followed by an **in-depth explanation** of what it does and why it is important.

---

## 1️⃣ Import Pandas

```python
import pandas as pd
```
# Explanation:
pandas is the primary Python library for working with structured data (CSV files, Excel sheets, SQL tables).
We import it as pd (common shorthand) to make the code easier to read.
Pandas provides the DataFrame, which is like a table in Excel or a SQL database — the main structure used in data science.

## 2️⃣ Load Datasets
Melbourne Housing Dataset:
```python
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
```

# Explanation:
pd.read_csv() loads CSV files into a Pandas DataFrame.
melbourne_data now holds all the rows and columns from the dataset.
Storing the file path in a variable makes your code cleaner and easier to update if the file location changes.

Ames Housing Dataset:
```python
train_file = '../input/ames-housing/train.csv'
train_data = pd.read_csv(train_file)
```

# Explanation:
same principle as above: loads the Ames housing CSV file into a DataFrame.
train_data can now be explored, summarized, and cleaned.
Having multiple datasets allows practice on different real-world scenarios.

## 3️⃣ Preview the Data
```python
melbourne_data.head()
train_data.head()
```

#Explanation:
.head() displays the first 5 rows of the DataFrame by default.
Helps verify that the dataset loaded correctly.
Gives a quick look at the column names, data types, and sample values.
Essential for spotting obvious data issues early.

## 4️⃣ Summary Statistics
```python
melbourne_data.describe()
train_data.describe()
```

# Explanation:
.describe() summarizes numerical columns only.
It outputs:
count → number of non-missing entries
mean → average value
std → standard deviation (spread of values)
min → smallest value
25%, 50%, 75% → percentiles
max → largest value
Helps understand distributions, detect outliers, and get a feel for typical values.

## 5️⃣ Check for Missing Values
```python
melbourne_data.isnull().sum()
train_data.isnull().sum()
```

# Explanation:
.isnull() identifies missing values in the DataFrame.
.sum() counts the number of missing values per column.
Missing values can affect model training and predictions.
Early detection allows planning cleaning or imputation strategies.

## 6️⃣ Inspect Column Types
```python
melbourne_data.dtypes
train_data.dtypes
```

# Explanation:
.dtypes shows the type of data in each column (e.g., int64, float64, object).
Important for:
Identifying numerical vs categorical features
Selecting appropriate preprocessing and models

## 7️⃣ Select Only Numerical Columns (Optional)
```python
numerical_data = melbourne_data.select_dtypes(include=['int64', 'float64'])
numerical_data.head()
```

# Explanation:
.select_dtypes() filters columns by data type.
Only numerical columns are selected here.
Useful for tasks like correlation analysis or model training.
Helps focus on features suitable for regression or other numerical predictions.

## ✅ Reflection Questions

Which columns might be useful for predicting house prices?

Are there any columns with too many missing values that might need cleaning?

Can you identify categorical vs numerical columns? Why is this important?

How does understanding the data at this stage improve model quality?







