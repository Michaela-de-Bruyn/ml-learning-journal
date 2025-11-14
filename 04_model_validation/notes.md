## Lesson 04: Model Validation

# Instructions / Summary

In this lesson, you will learn:

What model validation is and why it matters

Why evaluating a model on the same data it was trained on is a big mistake

What Mean Absolute Error (MAE) is and how to calculate it

How to properly evaluate a model using training and validation data

How to split data using train_test_split

How model accuracy changes when evaluated on data the model has not seen

This lesson introduces the correct way to measure predictive accuracy, a core skill for every data scientist.

# What is Model Validation?

To build good machine learning models, you need to evaluate how accurate they are. In most cases, the most important measure of model quality is predictive accuracy — how close the predictions are to the actual values.

A common beginner mistake is:

# Measuring predictive accuracy using the training data.

This leads to overly optimistic results. You’ll soon see why.

# Measuring Model Quality: Mean Absolute Error (MAE)

To evaluate a model, we need a single number that summarizes accuracy.
One commonly used metric is Mean Absolute Error (MAE).

# Formula

For a single prediction:

``` python 
error = actual - predicted
```


For MAE:

MAE = average of the absolute values of all errors
“On average, our predictions are off by about X.”

# Example Code (from the hidden model):
``` python
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
```

# Output:

``` 434.71594577146544 ```


This means the model's predictions were off by $434 on average — VERY GOOD, but misleading (you’ll see why next).

# The Problem with “In-Sample” Scores

The MAE we just calculated is based on:

Training the model on all data, and

Evaluating it using the same data

This is called an in-sample score, and it is misleading.

# Why is this bad?

Suppose in your small dataset:

All expensive houses happen to have green doors

But in the real world, door color has NOTHING to do with price

Your model will learn:

“Green door = expensive”

It will perform well on the training data, because that pattern appears there.
But in real-world data, the pattern won’t hold, and the model will fail.

This is why we must evaluate models using new data the model has never seen.

# The Solution: Train/Test Split

We fix this problem by splitting data into:

Training data → used to build the model

Validation data → used to test the model

We can do this easily with scikit-learn.

# Coding Model Validation
``` python
from sklearn.model_selection import train_test_split

# Split data into training and validation data
# random_state ensures you get the same split every time for consistency
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Define the model
melbourne_model = DecisionTreeRegressor()

# Fit the model on the training data
melbourne_model.fit(train_X, train_y)

# Make predictions on validation data
val_predictions = melbourne_model.predict(val_X)

# Calculate MAE on validation data
print(mean_absolute_error(val_y, val_predictions))
```


# Output:

``` 265806.91478373145 ```

In-sample MAE: ~ $500

Out-of-sample (realistic) MAE: ~ $265,000

This model performs extremely badly in the real world, even though it looked excellent on training data.

Why?
Because decision trees tend to overfit when not limited, memorizing the training data.

The validation error (~$265K) is about a quarter of the average home price in the dataset — far too high.

To improve this model, we could:

Try better features

Tune model parameters

Use different model types (Random Forests, Gradient Boosting, etc.)

Those improvements are coming in future lessons.

# Key Takeaways

Never trust in-sample accuracy.
Always evaluate on data that was not used to train the model.

MAE (Mean Absolute Error) is a useful and interpretable evaluation metric.

train_test_split is the standard method for performing model validation.

A model can appear to perform perfectly during training while being unusable in real-world scenarios.

Model validation ensures your model can generalize, not just memorize.
