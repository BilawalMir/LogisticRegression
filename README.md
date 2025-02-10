# Titanic Survival Prediction

This project uses machine learning to predict passenger survival on the Titanic based on various features.  It uses the Titanic dataset and builds a logistic regression model.

## Project Overview

This notebook walks through the process of:

1.  **Data Loading and Exploration:** Loads the Titanic dataset and explores its basic properties, like shape, data types, and missing values.
2.  **Data Cleaning:** Handles missing data by imputing age values based on passenger class and removing irrelevant columns like 'Ticket', 'Cabin', and 'Name'.
3.  **Data Preprocessing:** Converts categorical features like 'Sex' and 'Embarked' into numerical representations using one-hot encoding.  Drops the 'PassengerId' column as it's not relevant for prediction.
4.  **Model Training:** Trains a logistic regression model on the prepared data.
5.  **Model Evaluation:** Evaluates the model's performance using metrics like accuracy.

## Requirements

To run this notebook, you'll need the following libraries installed:

*   `numpy`
*   `pandas`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`

You can install these using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn

# Dataset
The dataset used in this project is the Titanic dataset.  It should be placed in the same directory as the notebook or provide the correct path to the CSV file. The code assumes the file is named titanic_dataset.csv.

## Running the Code
Make sure you have the required libraries installed.
Place the titanic_dataset.csv file in the correct location or modify the file path in the notebook.
Run the Jupyter Notebook cells sequentially.
## Code Explanation
The code performs the following key steps:

Imports: Imports necessary libraries for data manipulation, visualization, and machine learning.
Data Loading: Reads the Titanic dataset into a pandas DataFrame.
Data Cleaning:
Removes the 'Ticket', 'Cabin', and 'Name' columns as they are not used in the model.
Imputes missing 'Age' values based on the passenger's 'Pclass' (passenger class). The median age for each class is used for imputation.
Removes rows with any remaining missing values.
Data Preprocessing:
Converts the 'Sex' and 'Embarked' columns to categorical data types.
Uses one-hot encoding to convert categorical features into numerical representations.
Removes the original 'Sex' and 'Embarked' columns.
Removes the 'PassengerId' column.
Data Splitting: Splits the data into training and testing sets.
Model Training: Creates and trains a logistic regression model using the training data.
Model Prediction: Uses the trained model to make predictions on the test data.
Model Evaluation: Calculates and prints the accuracy of the model. It also prints the probabilities of survival for each passenger in the test set.
## Results
The output of the notebook will include the accuracy of the logistic regression model on the test set.  It will also show the predicted survival probabilities for each passenger in the test set.

## Conclusion
This project demonstrates a basic approach to predicting Titanic survival using logistic regression.  Further improvements could be made by exploring other machine learning models, feature engineering, and hyperparameter tuning.