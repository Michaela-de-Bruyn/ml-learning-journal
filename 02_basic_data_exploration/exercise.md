# Exercise: Average Lot Size & Newest Home Age

## Instructions
1. Load the Iowa housing dataset into a Pandas DataFrame.  
2. Calculate the **average lot size** (rounded to the nearest integer).  
3. Calculate the **age of the newest home** based on the year it was built.  
4. Use the provided `step_2.check()` function to verify your answers.

---

## Solution Code

```python
import pandas as pd

# Load the dataset
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)

# 1️⃣ What is the average lot size (rounded to nearest integer)?
avg_lot_size = round(home_data['LotArea'].mean())

# 2️⃣ As of today, how old is the newest home (current year - the date in which it was built)
current_year = 2025  # Replace with the current year if different
newest_home_age = current_year - home_data['YearBuilt'].max()

# 3️⃣ Checks your answers using Kaggle's step function
step_2.check()
```
# Explanation of Each Line:

```import pandas as pd```

Imports the Pandas library, which is essential for working with tabular data.

```pd``` is shorthand so code is easier to read.

```iowa_file_path = ...```

Stores the file path of the CSV dataset in a variable.

Using a variable makes it easy to reuse or change the path later.

```home_data = pd.read_csv(iowa_file_path)```

Reads the CSV into a Pandas DataFrame called home_data.

Now you can perform analysis, summary statistics, and calculations on this dataset.

```avg_lot_size = round(home_data['LotArea'].mean())```

```home_data['LotArea']``` selects the LotArea column.

```.mean()``` calculates the average lot size.

```round()``` rounds it to the nearest integer for readability.

```current_year = 2025```

Stores the current year in a variable.

Used to calculate the age of the newest home.

```newest_home_age = current_year - home_data['YearBuilt'].max()```

```home_data['YearBuilt'].max()``` finds the newest home’s year.

Subtracting from current_year gives its age.

```step_2.check()```

Kaggle helper function that checks your answers for correctness.

Ensures you calculated the average lot size and newest home age correctly.

# ✅ Key Takeaways

Using Pandas makes it easy to explore and summarize datasets.

Always assign file paths and results to variables for cleaner, reusable code.

Separating calculations (mean, max, subtraction) helps you understand each step.

Checking answers with Kaggle’s helper functions reinforces learning and builds confidence.
