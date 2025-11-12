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
```python
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
```

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

## Step 3: Select the prediction target
```python
# Prediction target (the variable to predict)
y = melbourne_data.Price
```

# Explanation:
y is our target variable, i.e., the column we want to predict.
Using dot notation (melbourne_data.Price) extracts the column as a Series.

## Step 4: Select features
```python
## # Define the features we want to use
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# Select features from the dataset
X = melbourne_data[melbourne_features]

# Review the feature data
X.describe()
X.head()
```

# Explanation:
melbourne_features is a list of column names we’ll use as input features.
X = melbourne_data[melbourne_features] creates a DataFrame with only the selected features.
.describe() gives summary statistics; .head() shows the first 5 rows.

## Step 5: Fit a Decision Tree model
```python
from sklearn.tree import DecisionTreeRegressor

# Define the model
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit the model with our features (X) and target (y)
melbourne_model.fit(X, y)
```

# Explanation:
DecisionTreeRegressor is a type of machine learning model for predicting numeric values.
random_state=1 ensures reproducible results.
.fit(X, y) trains the model to capture patterns in the data.

## Step 6: Make predictions
```python
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are:")
print(melbourne_model.predict(X.head()))
```
# Example Output:
Making predictions for the following 5 houses:
   Rooms  Bathroom  Landsize  Lattitude  Longtitude
1      2       1.0     156.0   -37.8079    144.9934
2      3       2.0     134.0   -37.8093    144.9944
4      4       1.0     120.0   -37.8072    144.9941
6      3       2.0     245.0   -37.8024    144.9993
7      2       1.0     256.0   -37.8060    144.9954

The predictions are:
[1035000. 1465000. 1600000. 1876000. 1636000.]

# Explanation:
.predict() uses the trained model to predict house prices for the first 5 rows.
The predictions are numerical estimates of home prices based on the features selected.

## ✅ Key Takeaways

Selecting target and features is crucial before building any model.

Dropping missing values is a simple way to clean data initially.

Reviewing data with .describe() and .head() helps detect anomalies early.

Fitting and predicting with a Decision Tree shows the full ML workflow: define → fit → predict → evaluate.
