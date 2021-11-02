#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 18:25:08 2021

@author: danielpicard
"""

## Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import train_test_split

## Data cleaning
loan_data = pd.read_csv('loans_full_schema.csv')

i = 0
for r in loan_data.isnull().any(axis=1):
    if r == True:
        i += 1
print('There are', i, 'rows with missing values') #Check for missing values as they will be problematic for modeling

print(loan_data.columns[loan_data.isnull().any() == True].tolist()) #Columns with missing values - will drop from df
loan_data = loan_data.drop(columns = ['emp_title', 'emp_length', 'debt_to_income', 'annual_income_joint', 'verification_income_joint', 'debt_to_income_joint', 'months_since_last_delinq', 'months_since_90d_late', 'months_since_last_credit_inquiry', 'num_accounts_120d_past_due'])

i = 0
for r in loan_data.isnull().any(axis=1):
    if r == True:
        i += 1
print('There are', i, 'rows with missing values') #Confirm missing values have been dealt with

## Describing data
print(loan_data.describe(include='all')) #Descriptive statistics for each column in dataset

## Visualizations

# Scatter of interest rate vs income
fig = plt.figure()
plot1 = sns.regplot(y = 'interest_rate', x = 'annual_income', scatter_kws = {'color' : 'grey'}, line_kws = {'color' : 'blue'}, data = loan_data)
plt.show(plot1)

# Scatter of total utilization % vs interest rate
loan_data['total_utilization'] = loan_data['total_credit_utilized'] / loan_data['total_credit_limit'] * 100
plot2 = sns.regplot(y = 'interest_rate', x = 'total_utilization', scatter_kws = {'color' : 'grey'}, line_kws = {'color' : 'blue'}, data = loan_data)
plt.show(plot2)

# Line graph of years since first credit line vs interest rate
loan_data['yrs_of_credit'] = 2021 - loan_data['earliest_credit_line']
yrs = loan_data.groupby('yrs_of_credit').mean()
line1 = plt.plot(yrs.index, yrs['interest_rate'])
plt.show(line1)

# Bar plot of average total balance by loan purpose
fig = plt.figure(figsize = (16, 10))
purpose = loan_data.groupby('loan_purpose').mean()
bar1 = plt.bar(purpose.index, purpose['balance'])
plt.show(bar1)

# Bar plot of average interest rate by loan grade
fig = plt.figure(figsize = (11, 10))
grade = loan_data.groupby('sub_grade').mean()
bar2 = plt.bar(grade.index, grade['interest_rate'])
plt.show(bar2)

## Modeling

loan_data = loan_data.dropna() #Drop rows with NaN values

#Create feature set by dropping target variable and non-numeric columns
feature_set = loan_data.drop(columns = ['interest_rate', 'state', 'homeownership', 'verified_income', 'loan_purpose', 'application_type', 'grade', 'sub_grade', 'issue_month', 'loan_status', 'initial_listing_status', 'disbursement_method'])
ys = loan_data['interest_rate']

for c in feature_set.columns:
    feature_set[c] = feature_set[c].astype(int)

#Split into testing and training
X_train, X_test, y_train, y_test = train_test_split(feature_set, ys, test_size = 0.25, random_state = 1)

lasso = LassoCV(cv=10, normalize=True, max_iter=10000) # Instantiate lasso regression object

ridge = RidgeCV(cv=10, normalize=True) # Instantiate ridge regression object

# Fit into one ensemble model
ensemble = VotingRegressor(estimators = [('r', ridge), ('l', lasso)]).fit(X_train, y_train)
predicted = ensemble.predict(X_test)

# Visualize results
plt.figure()
plt.plot(predicted, label='Predicted values for interest rate')