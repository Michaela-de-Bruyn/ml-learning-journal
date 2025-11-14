# Random Forests
# Instructions / Summary

In this lesson, you will learn:

Why decision trees face a trade-off between underfitting and overfitting

What a Random Forest is and why it performs better

How Random Forests combine many trees to produce more accurate predictions

How to build a RandomForestRegressor using scikit-learn

How to evaluate a random forest model using Mean Absolute Error (MAE)

Random Forests are one of the most powerful and easy-to-use models, offering strong performance with very little tuning.

# Why Random Forests?

Decision trees have a limitation:

Shallow Tree → Underfitting

Doesn’t capture enough patterns.

Deep Tree → Overfitting

Memorizes training data.

Even advanced models face this balance between simplicity and complexity.

# The Random Forest solution:

A random forest builds many decision trees, each trained slightly differently.
Its prediction is the average of all trees’ predictions.
This greatly reduces overfitting and usually improves accuracy.

Random forests:

✔ Reduce model variance
✔ Are more stable than single trees
✔ Work well with default settings
✔ Are less sensitive to hyperparameters

# How Random Forests Work (Simplified)

A random forest trains multiple trees where each tree:

Sees a random subset of rows

Sees a random subset of features

Produces a prediction

Final prediction = average of all tree predictions

This averaging smooths out the noise from any single tree.

# Example: Building a Random Forest

We assume your data is already split as:

```train_X```

```val_X```

```train_y```

```val_y```

# Solution Code:
``` python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Define the random forest model
forest_model = RandomForestRegressor(random_state=1)

# Fit the model on the training data
forest_model.fit(train_X, train_y)

# Make predictions using validation data
melb_preds = forest_model.predict(val_X)

# Evaluate using MAE
print(mean_absolute_error(val_y, melb_preds))
```

# Expected Output: 
```191669.7536453626```

# Interpretation

Best decision tree MAE was around 250,000

Random forest MAE is ~191,670

This is a MAJOR improvement in accuracy.

And importantly:

Random Forests perform well even without tuning hyperparameters.
This makes them an excellent first-choice model for many real-world problems.

# Conclusion

Random Forests solve many of the problems that decision trees struggle with:

✔ Reduce overfitting
✔ Improve prediction accuracy
✔ Require minimal tuning
✔ Work well on large and complex datasets

While more advanced models exist (XGBoost, LightGBM, CatBoost, Neural Networks), Random Forests provide a strong foundation and a reliable first choice for many machine learning tasks.

# Key Takeaways

A single decision tree is simple but prone to underfitting or overfitting.

Random Forests combine many trees to produce more stable and accurate predictions.

They work extremely well with default settings.

Evaluate performance using Mean Absolute Error (MAE).

Random Forests often outperform individual decision trees by a large margin.

