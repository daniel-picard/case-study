#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 20:15:31 2021

@author: danielpicard
"""

# Imports

import pandas as pd
import matplotlib.pyplot as plt

# Analysis

cust_data = pd.read_csv('casestudy.csv')

def descInfo(year):
    
    current_customers = cust_data[cust_data['year'] == int(year)]
    
    prev_customers = cust_data[cust_data['year'] == int(year) - 1]
    
    minus2_customers = cust_data[cust_data['year'] == int(year) - 2]
    
    diff = current_customers.append(prev_customers)
    diff2 = diff.drop_duplicates(subset = ['customer_email'], keep=False)
    
    new_customers = diff2[diff2['year'] == int(year)]
    lost_customers = diff2[diff2['year'] == int(year) - 1]
        
    #Display calculated values
    
    print('YEAR:', year)
    
    print('\nTotal Revenue for Current Year:', sum(current_customers['net_revenue']))
    print('Total Revenue from New Customers:', sum(new_customers['net_revenue']))
    print('Existing Customer Growth:', sum(current_customers['net_revenue']) - sum(new_customers['net_revenue']) - sum(prev_customers['net_revenue']))
    print('Revenue Lost to Attrition:', sum(lost_customers['net_revenue']))
    print('Existing Customer Revenue Current Year:', sum(current_customers['net_revenue']) - sum(new_customers['net_revenue']))
    print('Existing Customer Revenue Current Year:', sum(prev_customers['net_revenue']) - sum(minus2_customers['net_revenue']))
    print('Total Customers Current Year:', len(current_customers))
    print('Total Customers Previous Year:', len(prev_customers))
    print('New Customers:', len(new_customers))
    print('Lost Customers:', len(lost_customers))
    
descInfo('2016')

# Visualizations
plt.figure(figsize=(8, 8))
by_yr = cust_data.groupby('year').sum()
plt.bar(by_yr.index, by_yr['net_revenue']) #Line graph of total revenue per year
plt.show()

plt.figure(figsize=(8, 8))
by_yr = cust_data.groupby('year').mean()
plt.bar(by_yr.index, by_yr['net_revenue']) #Line graph of average revenue by customer per year
plt.show()