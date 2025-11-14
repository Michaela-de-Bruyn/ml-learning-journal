## Exercise: Underfitting & Overfitting (Lesson 05)
# Instructions / Summary

In this exercise you will:

Compare different decision tree sizes

Measure their performance using Mean Absolute Error (MAE)

Identify the optimal tree size

Fit a final model using the best tree size and all available data

This helps you practice finding the sweet spot between underfitting and overfitting.

# Setup Code: 
``` python
# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)

# Create target object (y)
y = home_data.SalePrice

# Create feature set (X)
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
            'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into training and validation datasets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify the model
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit model
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate MAE
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE: {:,.0f}".format(val_mae))

# Setup code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex5 import *
print("\nSetup complete")
```

# Helper Function (Provided)
``` python
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```

This function:

Builds a tree with a specific ```max_leaf_nodes```

Trains it using training data

Predicts using validation data

Returns the MAE

# Step 1: Compare Different Tree Sizes
Instructions

Try each of these values for```max_leaf_nodes:```
``` python

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
```

Use a loop to:

Call ```get_mae()``` on each value

Store the MAEs

Identify the tree size that gives the lowest MAE

Save it to ```best_tree_size```

# Solution Code:
``` python
# Step 1: Compare Different Tree Sizes

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

# Track the best tree size and lowest MAE
scores = {}
for leaf_size in candidate_max_leaf_nodes:
    mae = get_mae(leaf_size, train_X, val_X, train_y, val_y)
    scores[leaf_size] = mae

# Find the tree size with the lowest MAE
best_tree_size = min(scores, key=scores.get)

# Check your answer
step_1.check()
```

# Explanation: 

A loop tests each candidate tree size

Each MAE is stored in a dictionary

```min(scores, key=scores.get)``` returns the key with the lowest error

This is the best tree size

# Step 2: Fit the Final Model Using All Data

Once the best tree size is known, you should:

Train a new model using all data (X, y)

Do NOT use validation split here

Use the chosen ```max_leaf_nodes``` to avoid overfitting

# Solution Code:
``` python
# Step 2: Fit Model Using All Data

# Create final model using optimal leaf size
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

# Fit on ALL data (no train/val split)
final_model.fit(X, y)

# Check your answer
step_2.check()
```

# Explanation: 

After selecting the best tree size, you rebuild the model on all available data

This improves accuracy because no data is held out

Now the model is ready for real-world use

# Key Takeaways

Decision trees must be tuned to avoid underfitting or overfitting

```max_leaf_nodes``` is an effective way to control tree complexity

Testing different tree sizes helps you find the optimal model

Always choose based on validation MAE, not training MAE

Once the best model is found, retrain on all the data before deployment
