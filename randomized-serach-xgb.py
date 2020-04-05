# -----------------------------------------
# Hyperparameter tuning of XGBoost
# 
# Date: 2020-03-13
# -----------------------------------------


#%% -----------------------------------------
# Step 1 - Data engineering
# -------------------------------------------


#%% -----------------------------------------
# Import packages
# -------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import os
import time

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb


#%% -----------------------------------------
# Settings
# -------------------------------------------

# Plot settings
sns.set()


#%% -----------------------------------------
# Load data
# -------------------------------------------

# Home credit default data from Kaggle
app_train = pd.read_csv('data/application_train.csv')
app_test = pd.read_csv('data/application_test.csv')

# Make copies to modify
train_df = app_train.copy()
validation_df = app_test.copy()


#%% -----------------------------------------
# Functions
# -------------------------------------------

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        # Percentage of missing values
        mis_val_percent = 100*df.isnull().sum()/len(df)
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent, df.dtypes], axis=1)
        # Rename the columns
        mis_val_table.columns = ['nr_null', 'share', 'dtype']
        # Sort the table by percentage of missing descending
        mis_val_table = mis_val_table[mis_val_table.iloc[:,1] != 0].sort_values(by=['share'], ascending=False).round(1)
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table

# Characteristics table
def char_table_fn(x):
    res_df = pd.DataFrame()
    for i in x.select_dtypes(['float', 'int']).columns:
        res_df = res_df.append(pd.DataFrame(x[i].describe()).T)
        res_df.loc[i, 'median'] = x[i].median()
        res_df.loc[i, 'median_mean_dev'] = np.abs((res_df.loc[i, 'median'] - res_df.loc[i, 'mean'])/res_df.loc[i, 'median'])

    return(res_df.sort_values('median_mean_dev', ascending=False))



#%% -----------------------------------------
# Data engineering
# -------------------------------------------

# Initial info about our dataset
train_df.info()

# Target variable 
train_df['TARGET'].describe()

# Balance of target variable
train_df['TARGET'][train_df['TARGET'] != 1.0].count()/train_df['TARGET'][train_df['TARGET'] == 1].count()
"""
sum(negative instances)/sum(positive instances) = 11.38
Fairly unbalanced data.
"""

# # Drop uuid column from both tables
# train_df = train_df.drop(columns=['uuid'])
# test_df = test_df.drop(columns=['uuid'])

# ----------------------------

# Check missing values
miss_val_df = missing_values_table(train_df)
miss_val_df.head(15)
"""
All missing columns with missing values are floats.
"""

# Columns types
train_df.dtypes.value_counts()

# Unique classes in each object type
train_df.select_dtypes('object').apply(pd.Series.nunique, axis=0)
validation_df.select_dtypes('object').apply(pd.Series.nunique, axis=0)
"""
We apply one-hot encoding to all categorical features
"""

# ----------------------------

# One-hot encoding
ohe = OneHotEncoder(handle_unknown='ignore')

ohe_df = train_df.select_dtypes('object')
ohe_column_names = ohe_df.columns

# Fill missing values
ohe_df = ohe_df.fillna('missing')

ohe.fit(ohe_df)

# Apply to train data
cat_feat_train = ohe.transform(ohe_df)
cat_feat_train = pd.DataFrame(
    cat_feat_train.todense(),
    columns=list(ohe.get_feature_names(ohe_column_names))
)

# Apply to test data
cat_feat_test = ohe.transform(validation_df[ohe_column_names].fillna('missing'))
cat_feat_test = pd.DataFrame(
    cat_feat_test.todense(),
    columns=list(ohe.get_feature_names(ohe_column_names))
)

# Check number of features
print('Training OHE feature shape: ', cat_feat_train.shape)
print('Test OHE feature shape:     ', cat_feat_test.shape)
"""
Both OHE training data and test data has the same number
of columns.
"""

# ----------------------------

# Assign new dummy columns to original datasets
train_df = train_df.drop(columns=ohe_column_names)
train_df = pd.concat([train_df, cat_feat_train], axis=1)

validation_df = validation_df.drop(columns=ohe_column_names)
validation_df = pd.concat([validation_df, cat_feat_test], axis=1)

# Check features of train and test dataset
print('Training feature shape: ', train_df.shape)
print('Test feature shape:     ', validation_df.shape)

"""
We stop the data engineering part here.

In order to arrive at a better model performance, exploratory analysis and feature engineering
should take place here before we train our model.
"""


#%% -----------------------------------------
# Data prep
# -------------------------------------------

# Assign train datasets with new names
X = train_df.drop(columns=['TARGET']).copy()
X_validation = validation_df.copy()
y = train_df['TARGET']

# Shape of data
print('The shape of X is:                ', X.shape)
print('The shape of X_last_validation is:', X_validation.shape)

# Missing values
X_missing_values = missing_values_table(X)
"""
All variables with missing values are floats.
"""

# Check nr unique values for columns with missing values
X[X_missing_values.index].apply(pd.Series.nunique, axis=0).sort_values()

# Impute missing values with median
imputer_median = SimpleImputer(strategy='median')
imputer_median.fit(X)

X = imputer_median.transform(X)

# Convert to dataframe
X = pd.DataFrame(X, columns=train_df.drop(columns=['TARGET']).columns)

# Check
missing_values_table(X)
"""
Seem to work
"""

# Same procedure for our validation data
X_validation = imputer_median.transform(X_validation)

# Convert to dataframe
X_validation = pd.DataFrame(X_validation, columns=validation_df.columns)

# Check
missing_values_table(X_validation)

# ----------------------------

# Split data into training and test sets
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# ----------------------------

#%% -----------------------------------------
# XGBoost
# -------------------------------------------

"""
XGBoost can handle imbalanced data through the parameter scale_pos_weight
A typical value to consider: sum(negative instances) / sum(positive instances)
"""

bal = y_train[y_train != 1.0].count()/y_train[y_train == 1.0].count()

# --------------------------
# Naive approach
# --------------------------

"""
Here we use one set of parameters and produce a XGBoost model
"""

# XGBoost classifier
cl_xgb = xgb.XGBClassifier(
    max_depth=4,
    learning_rate=0.07,
    n_estimators=500,
    colsample_bytree=0.3,
    scale_pos_weight=bal,
    reg_alpha=4,
    random_state=123
).fit(X_train, y_train)

# Feature importance
plt.figure(figsize=(10, 7))
sns.barplot(
    x=pd.Series(cl_xgb.get_booster().get_score(), index=[i for i in cl_xgb.get_booster().get_score()]).sort_values(ascending=False).iloc[:20].values,
    y=pd.Series(cl_xgb.get_booster().get_score(), index=[i for i in cl_xgb.get_booster().get_score()]).sort_values(ascending=False).iloc[:20].index,
    color='b')
plt.title('Feature importance top 20 predictors, XGBoost - naive approach')
plt.xlabel('F1 score')

# Print results
print('Accuracy score for XGBoost classifier:', accuracy_score(y_test, cl_xgb.predict(X_test)))
print('ROC AUC score for XGBoost classifier: ', roc_auc_score(y_test, cl_xgb.predict(X_test)))
print('F1 score for XGBoost classifier:      ', f1_score(y_test, cl_xgb.predict(X_test)))


# --------------------------
# Randomized search
# --------------------------

# Hyper-parameters
params_cv = {
    'colsample_bytree': [0.3, 0.5, 0.8],                # subsample ratio of columns constructing each tree
    'learning_rate': np.arange(0.03, 0.10, step=0.02),  # step size shrinkage 
    'max_depth': np.arange(3, 6),                       # max depth of tree
    'n_estimators': np.arange(200, 600, step=50),       # number of trees used
    'subsample': [0.5, 0.8, 1.0],                       # subsample ratio of training instances
    'min_child_weight': [5, 8, 10, 15],                 # minimum sum of instance weight needed in a child
    'alpha': [0, 1, 2, 3, 4, 6],                        # L2 regularization term
    'gamma': [0, 1, 2, 3, 4, 6],                        # minimum loss reduction for further partition
    'max_delta_step': [0, 1, 2, 3, 4]                   # maximum step each leaf output must be (good for imblanaced data)
}

# Model
xgb_cv_model = xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=bal, random_state=123)

# Randomized search
xgb_cv_rs = RandomizedSearchCV(
    estimator=xgb_cv_model,
    param_distributions=params_cv,
    n_iter=7,
    scoring='f1',
    n_jobs=-1,
    iid=True,
    cv=5,
    verbose=1
)

# Run search
print('Randomized search - Parameter tuning')
tic = time.time()
xgb_cv_rs.fit(X_train, y_train)
toc = time.time()
print('Total elapsed time:', (toc-tic)/60, 'minutes.')

# Best parameters
print('Best parameters from random search:\n', xgb_cv_rs.best_params_)

# Results
print('Accuracy score for XGBoost classifier:', accuracy_score(y_test, xgb_cv_rs.best_estimator_.predict(X_test)))
print('ROC AUC score for XGBoost classifier: ', roc_auc_score(y_test, xgb_cv_rs.best_estimator_.predict(X_test)))
print('F1 score for XGBoost classifier:      ', f1_score(y_test, xgb_cv_rs.best_estimator_.predict(X_test)))


#%% -----------------------------------------
# Apply on final validation data
# -------------------------------------------

# Compute TARGET and PD using the XGBoost model
validation_df['TARGET'] = xgb_cv_rs.best_estimator_.predict(X_validation)
validation_df['PD'] = xgb_cv_rs.best_estimator_.predict_proba(X_validation)[:,1]

# Print results
print(validation_df['SKID', 'TARGET', 'PD'])