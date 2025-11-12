# Lesson 2: Basic Data Exploration

## Objective
Learn how to explore and understand your data using Python's Pandas library before building machine learning models.

## Key Concepts
- **Pandas**: Python library for data manipulation and analysis.
- **DataFrame**: Main data structure in Pandas, similar to Excel or SQL tables.
- **describe()**: Summarizes numerical columns (count, mean, std, min, 25%, 50%, 75%, max)
- **Missing Values**: Columns may have missing data; understanding them is crucial before modeling.
- **Target Variable**: The variable we want to predict (e.g., `SalePrice` in a housing dataset).

## Example Datasets

### 1. Melbourne Housing Dataset
- File: `melb_data.csv`  
- Columns include: Rooms, Price, Distance, Bedroom2, Bathroom, Car, Landsize, BuildingArea, YearBuilt, Lattitude, Longtitude, Propertycount  
- Source: [Kaggle – Melbourne Housing Snapshot](https://www.kaggle.com/dansbecker/basic-data-exploration)

### 2. Ames Housing Prices Competition
- Files: `train.csv`, `train.csv.gz`, `test.csv`, `test.csv.gz`, `sample_submission.csv`, `data_description.txt`  
- Target variable: `SalePrice`  
- Source: Kaggle competition – [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

### 3. Mobile Price Classification Dataset
- Files: `train.csv`, `test.csv`  
- Target: mobile price range class  
- Source: Kaggle – search for “Mobile Price Classification”

## Key Notes
- Explore columns to understand distributions and potential issues.
- Check for missing values and inconsistencies.
- Identify which columns may be useful for predictive modeling.
- Understanding your data is the first step before training a machine learning model.
