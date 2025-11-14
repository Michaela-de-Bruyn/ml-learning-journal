## Exercise: Model Validation (Lesson 04)
Instructions / Summary

In this exercise, you will:

Split your dataset into training and validation sets.

Create and fit a Decision Tree Regressor using only the training data.

Make predictions using validation data.

Compare validation predictions to actual values.

Calculate Mean Absolute Error (MAE) to evaluate the model.

This exercise reinforces why in-sample predictions are misleading and why validation is essential for real-world machine learning.

``` python
# Code you have previously used to load data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 
                   'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex4 import *
print("Setup Complete")
```
# Step 1: Split Your Data
Instructions

Use ```train_test_split``` to divide data into:

```train_X, val_X```

```train_y, val_y```

Use ```random_state = 1```

# Solution Code:
``` python # Step 1: Split Your Data

# Import the train_test_split function
from sklearn.model_selection import train_test_split

# Split data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Check your answer
step_1.check()
```
# Explanation:

```train_test_split``` randomly splits your data into training and validation subsets.

```random_state=1``` ensures the same split each time you run the code.

```train_X``` and ```train_y``` → used to fit the model

```val_X``` and ```val_y``` → used only for evaluation

# Step 2: Specify and Fit the Model
Solution Code:
```python # Step 2: Specify and Fit the Model

# Specify the model with random_state for reproducibility
iowa_model = DecisionTreeRegressor(random_state=1)

# Fit the model with the training data
iowa_model.fit(train_X, train_y)

# Check your answer
step_2.check()
```

# Explanation:

A new ```DecisionTreeRegressor``` is created.

It is trained only on the training data.

This prevents the model from memorizing the validation set.

# Step 3: Make Predictions with Validation Data
Solution Code:
``` python
# Step 3: Make Predictions with Validation Data

# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)

# Check your answer
step_3.check()
```

# View top predictions vs actual prices
``` python
# Print the top few validation predictions
print(val_predictions[:5])

# Print the top few actual prices from validation data
print(val_y.head())
```

# Explanation:

These predictions will differ from the in-sample predictions shown earlier.

Why? Because the model never saw validation data during training.

This is a more honest evaluation of the model’s real-world performance.

# Step 4: Calculate the Mean Absolute Error (MAE)
Solution Code:
``` python
# Step 4: Calculate the Mean Absolute Error in Validation Data

from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y, val_predictions)

# Uncomment to print MAE
# print(val_mae)

# Check your answer
step_4.check()
```
# Explanation:

```mean_absolute_error()``` measures how far off the predictions are, on average.

MAE on validation data gives you a realistic estimate of model accuracy.

# Key Takeaways

In-sample accuracy is misleading.
A model may appear perfect when evaluated on training data.

Validation accuracy shows real performance.
It evaluates the model on data it hasn’t seen before.

MAE is a simple and intuitive metric for comparing model performance.

train_test_split is essential for proper model evaluation.

Always validate before deploying or trusting model predictions.
