# Exercise: Iowa Housing – Predicting Sale Price

## Instructions
1. Load the Iowa housing dataset into a Pandas DataFrame.
2. Select the target variable (SalePrice) for prediction.
3. Choose a subset of predictive features to create a DataFrame X.
4. Specify and fit a Decision Tree Regressor using X and y.
5. Make predictions on the training data and inspect the results.
6. Review the top predictions versus actual sale prices.
This exercise demonstrates the full machine learning workflow: define → fit → predict → evaluate.

# Setup
```python
# Import Pandas and set up dataset path
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Load the dataset
home_data = pd.read_csv(iowa_file_path)

# Set up Kaggle code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *

print("Setup Complete")
```

# Explanation:
```pandas``` is used for working with tabular data.
```home_data``` stores the dataset in a DataFrame.
Kaggle helper functions ```(binder.bind``` and ```step_X.check())``` allow checking your answers.

# Step 1: Specify Prediction Target
```python
# Print the list of columns to identify the prediction target
print(home_data.columns)
# Select the prediction target for the Iowa dataset
y = home_data.SalePrice

# Check your answer
step_1.check()
```

# Explanation:
```home_data.SalePrice``` selects the column representing sale price.
```step_1.check()``` verifies your target variable is correct.

# Step 2: Create X (Features)
```python
# Step 2: Create X (the features)

# Create the list of features
feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
                 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Select data corresponding to features
X = home_data[feature_names]

# Check your answer
step_2.check()
```

# Explanation:
```feature_names``` lists columns to include as features.
```X = home_data[feature_names]``` creates a DataFrame with only those features.
```step_2.check()``` confirms your selection is correct.

# Review Data
```python
# Review statistics and top rows
X.describe()  # Summary statistics
X.head()      # Top few rows
```

# Explanation:
```.describe()``` gives count, mean, min, max, and percentiles.
```.head()``` shows the first rows for visual inspection.

# Step 3: Specify and Fit Model
```python
# Step 3: Specify and Fit Model

# Import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

# Specify the model with random_state for reproducibility
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model using features X and target y
iowa_model.fit(X, y)

# Check your answer
step_3.check()
```

# Explanation:
```DecisionTreeRegressor(random_state=1)``` creates the model.
```.fit(X, y)``` trains it using the features and target variable.
```step_3.check()``` verifies the model is correctly fitted.

## Step 4: Make Predictions
```python
# Step 4: Make Predictions

# Use the fitted model to make predictions on X
predictions = iowa_model.predict(X)

# Print the predictions
print(predictions)

# Check your answer
step_4.check()
```

# Explanation:
```.predict(X)``` generates predictions for each home.
```predictions``` stores these values for evaluation.
```step_4.check()``` confirms the predictions are correctly generated.

# Compare Predictions vs Actual Values
```python
# Compare top predictions to actual sale prices
print(pd.DataFrame({'Predicted': predictions, 'Actual': y}).head())
```

# Explanation:
This lets you visually inspect how close the model's predictions are to the actual sale prices.

## Key Takeaways

Always define target ```(y)``` and features ```(X)``` before modeling.
A Decision Tree illustrates the ML workflow: define → fit → predict → evaluate.
```.describe()``` and ```.head()``` help detect anomalies before modeling.
Kaggle helper functions reinforce correct implementation and understanding.

