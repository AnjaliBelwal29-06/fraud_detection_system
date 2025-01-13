FRAUD DETECTION DATA PREPROCESSING:
This project involves data preprocessing for a machine learning model.
The dataset consists of transactions,with numeric columns having missing values that are imputed using the mean of each respective column. 
The data also contains categorical columns which are untouched in this process.

PROJECT OVERVIEW
Objective: Handle missing values in the dataset, especially for numeric columns, and prepare the data for further analysis or use in machine learning models.
Tech Stack:
Python
Pandas
Dataset
The dataset includes the following columns:

transaction_id: numeric.
transaction_amount: numeric.
location: string.
merchant: string.
Age: numeric.
gender: string.
fraud_label: numeric..

Installation
To run this project, you will need Python 3.x and the following dependencies:

To install the required dependencies, run the following:
#pip install pandas scikit-learn

Code Overview
Load Dataset: The dataset contains columns like Amount, Age, and TransactionType.
Handle Missing Values: Numeric columns (Amount, Age) are filled with their mean values.
Output:
  transaction_id  transaction_amount       location  merchant  age gender  fraud_label
0               1              1000.0       New York  ABC Corp   35      M            0
1               2               500.0        Chicago   XYZ Inc   45      F            0
2               3              2000.0    Los Angeles  ABC Corp   28      M            1
3               4              1500.0  San Francisco   XYZ Inc   30      F            0
4               5               800.0        Chicago  ABC Corp   50      F            0
Confusion Matrix:
[[24  0]
 [ 0  2]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        24
           1       1.00      1.00      1.00         2

    accuracy                           1.00        26
   macro avg       1.00      1.00      1.00        26
weighted avg       1.00      1.00      1.00        26


Accuracy: 1.00

