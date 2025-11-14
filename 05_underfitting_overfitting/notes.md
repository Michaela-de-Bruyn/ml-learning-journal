## Lesson 05: Underfitting & Overfitting
# Instructions / Summary

In this lesson, you will learn:

What underfitting means

What overfitting means

Why the depth of a decision tree affects model performance

How to use max_leaf_nodes to control model complexity

How to compare multiple models using Mean Absolute Error (MAE)

How to choose the best model using validation data

This lesson introduces one of the most important ideas in machine learning:
Finding the balance between Underfitting ↔ Overfitting

# Understanding Model Complexity

A decision tree becomes more complex as it adds more splits (i.e., becomes deeper).

![Shallow decision tree](images/shallow_tree.png)

<Image: shallow_tree.png>

Makes very few splits

Groups many houses together

Fails to capture important patterns

Performs poorly on training and validation data

An overly deep tree (overfitting)

Splits too many times

Learns small, random patterns specific to the training data

Performs extremely well on training data

Performs poorly on new or validation data

The goal

Find the sweet spot where the model is complex enough to capture real patterns,
but not so complex that it memorizes noise.

![Underfitting vs Overfitting curve](images/validation_curve.png)

This “sweet spot” is usually where the validation MAE is lowest.

# Controlling Model Complexity with ```max_leaf_nodes```

Decision trees can be constrained using:
``` python
max_depth  
min_samples_split  
min_samples_leaf  
max_leaf_nodes  ← Easiest and most useful for beginners
```

max_leaf_nodes controls how many terminal nodes (leaves) the tree may have.

Small number of leaves → underfitting

Large number of leaves → overfitting

# Comparing Models with Different Tree Sizes

We’ll create a helper function to calculate MAE for a given number of leaf nodes.

# Solution Code:
``` python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

# Utility function to calculate MAE for a given max_leaf_nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae
```
``` python
Compare MAE for Different Tree Sizes
# Compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %d" % (max_leaf_nodes, my_mae))
```

# Expected Output:
Max leaf nodes: 5            Mean Absolute Error: 347380
Max leaf nodes: 50           Mean Absolute Error: 258171
Max leaf nodes: 500          Mean Absolute Error: 243495
Max leaf nodes: 5000         Mean Absolute Error: 254983

# Interpretation

With 5 leaves, the model is too simple → underfitting

With 5000 leaves, the model memorizes noise → overfitting

With 500 leaves, MAE is the lowest → best balance

Optimal model: 500 leaf nodes

# Conclusion
# Overfitting

The model learns patterns that only exist in the training data.

Training accuracy = very high

Validation accuracy = very poor

Model is not useful in real world

# Underfitting

The model is too simple to capture real relationships.

Training accuracy = poor

Validation accuracy = poor

Good Fit

The model is just complex enough to capture useful patterns.

Validation MAE is lowest

Best model for real-world predictions

## Key Takeaways

Machine learning models must balance bias (underfitting) and variance (overfitting).

The depth of a decision tree strongly affects this balance.

Use validation data to evaluate how well your model generalizes to new data.

max_leaf_nodes is a simple and powerful way to control tree complexity.

The best model is the one with the lowest validation MAE, not the lowest training error.
