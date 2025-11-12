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

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Explanation:
pd.read_csv() loads CSV files into a Pandas DataFrame.
melbourne_data now holds all the rows and columns from the dataset.
Storing the file path in a variable makes your code cleaner and easier to update if the file location changes.

Ames Housing Dataset:
train_file = '../input/ames-housing/train.csv'
train_data = pd.read_csv(train_file)

# Explanation:
same principle as above: loads the Ames housing CSV file into a DataFrame.
train_data can now be explored, summarized, and cleaned.
Having multiple datasets allows practice on different real-world scenarios.

## 3️⃣ Preview the Data
melbourne_data.head()
train_data.head()

#Explanation:
.head() displays the first 5 rows of the DataFrame by default.
Helps verify that the dataset loaded correctly.
Gives a quick look at the column names, data types, and sample values.
Essential for spotting obvious data issues early.

## 4️⃣ Summary Statistics
melbourne_data.describe()
train_data.describe()

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
melbourne_data.isnull().sum()
train_data.isnull().sum()




