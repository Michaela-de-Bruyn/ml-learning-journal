# Exercise: Random Forests (Lesson 06)
# Instructions / Summary

Replace the Decision Tree with a Random Forest model.

Fit the model to the training data.

Predict on validation data and compute the Mean Absolute Error (MAE).

Compare the Random Forest MAE to the tuned Decision Tree MAE.

Random Forests often give a substantial performance boost with little or no hyperparameter tuning.

# Recap / Setup

Run this code cell to set up the environment exactly where the previous steps left off.
``` python 
# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y
y = home_data.SalePrice

# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Example: Decision tree you tuned earlier (for reference)
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(train_X, train_y)
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best Decision Tree (max_leaf_nodes=100): {:,.0f}".format(val_mae))

# Set up code checking (Kaggle helper)
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex6 import *
print("\nSetup complete")
```

# Output example:

``` Validation MAE for best Decision Tree (max_leaf_nodes=100): 243,495``` 
```Setup complete```

# Step 1: Use a Random Forest
# Instructions: 

Create a RandomForestRegressor (set random_state=1).

Fit it on train_X, train_y.

Predict on val_X and compute mean_absolute_error against val_y.

Save the MAE to rf_val_mae, print it, then run step_1.check().

# Solution Code:
``` python 
# Step 1: Use a Random Forest
from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# Fit the model on training data
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = mean_absolute_error(val_y, rf_model.predict(val_X))

print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# Check your answer
step_1.check()
```

Expected behavior:
You should see a printed MAE (numeric) and ```step_1.check()``` should confirm (if run in the learning environment).

# Explanation of Each Line
```from sklearn.ensemble import RandomForestRegressor```

Imports the Random Forest model class from scikit-learn.

```rf_model = RandomForestRegressor(random_state=1)```

Creates a Random Forest regressor instance.

```random_state=1``` ensures reproducible results.

```rf_model.fit(train_X, train_y)```

Trains the Random Forest on the training data (train_X, train_y).

Each tree in the forest is built using random subsets of rows and features.

```rf_val_mae = mean_absolute_error(val_y, rf_model.predict(val_X))```

```rf_model.predict(val_X)``` computes predicted sale prices for the validation set.

```mean_absolute_error(val_y, ...)``` computes the average absolute difference between actual and predicted prices; lower is better.

```print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))```

Nicely prints the MAE rounded for readability.

```step_1.check()```

Kaggle’s checking function to validate your solution (only available in the Kaggle lesson environment).

# Key Takeaways

Random Forests often yield better validation performance than a single Decision Tree because averaging many trees reduces variance/overfitting.

Random Forests typically work well with default parameters, making them an excellent first-choice model.

Always compare MAE (or other validation metrics) to judge the real improvement.
