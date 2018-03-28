# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:48:39 2018

@author: deepak
"""

Python 3.6.1 |Anaconda 4.4.0 (64-bit)| (default, May 11 2017, 13:25:24) [MSC v.1900 64 bit (AMD64)]
Type "copyright", "credits" or "license" for more information.

IPython 5.3.0 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.

runfile('C:/Users/deepak/nested function.py', wdir='C:/Users/deepak')

1
Out[2]: 1


def bar():
    x = 10
    def spam(): # Nested function definition
        print ('x is', x)
    while x > 0:
        spam()
        x -= 1
        

bar(3)
Traceback (most recent call last):

  File "<ipython-input-4-2f0fa4d9f487>", line 1, in <module>
    bar(3)

TypeError: bar() takes 0 positional arguments but 1 was given


bar()
x is 10
x is 9
x is 8
x is 7
x is 6
x is 5
x is 4
x is 3
x is 2
x is 1

def bar(a):
    def spam(): # Nested function definition
        print ('x is', x)
    while x > 0:
        spam()
        x -= 1
bar(10)

Traceback (most recent call last):

  File "<ipython-input-6-6e8c1837e8bf>", line 7, in <module>
    bar(10)

  File "<ipython-input-6-6e8c1837e8bf>", line 4, in bar
    while x > 0:

UnboundLocalError: local variable 'x' referenced before assignment


def bar(a):
    def spam(): # Nested function definition
        print ('x is', x)
    while x > 0:
        spam()
        x -= 1
        

bar(10)
Traceback (most recent call last):

  File "<ipython-input-8-695f6e2935b3>", line 1, in <module>
    bar(10)

  File "<ipython-input-7-de10be7b27f9>", line 4, in bar
    while x > 0:

UnboundLocalError: local variable 'x' referenced before assignment


def bar(x):
    def spam(): # Nested function definition
        print ('x is', x)
    while x > 0:
        spam()
        x -= 1
        

bar(10)
x is 10
x is 9
x is 8
x is 7
x is 6
x is 5
x is 4
x is 3
x is 2
x is 1

bar(45)
x is 45
x is 44
x is 43
x is 42
x is 41
x is 40
x is 39
x is 38
x is 37
x is 36
x is 35
x is 34
x is 33
x is 32
x is 31
x is 30
x is 29
x is 28
x is 27
x is 26
x is 25
x is 24
x is 23
x is 22
x is 21
x is 20
x is 19
x is 18
x is 17
x is 16
x is 15
x is 14
x is 13
x is 12
x is 11
x is 10
x is 9
x is 8
x is 7
x is 6
x is 5
x is 4
x is 3
x is 2
x is 1

clear


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn


from sklearn.datasets import load_boston

boston = load_boston()

boston.keys()
Out[16]: dict_keys(['data', 'target', 'feature_names', 'DESCR'])

boston()
Traceback (most recent call last):

  File "<ipython-input-17-3db1435b41e3>", line 1, in <module>
    boston()

TypeError: 'Bunch' object is not callable


boston
Out[18]: 
{'DESCR': "Boston House Prices dataset\n===========================\n\nNotes\n------\nData Set Characteristics:  \n\n    :Number of Instances: 506 \n\n    :Number of Attributes: 13 numeric/categorical predictive\n    \n    :Median Value (attribute 14) is usually the target\n\n    :Attribute Information (in order):\n        - CRIM     per capita crime rate by town\n        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n        - INDUS    proportion of non-retail business acres per town\n        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n        - NOX      nitric oxides concentration (parts per 10 million)\n        - RM       average number of rooms per dwelling\n        - AGE      proportion of owner-occupied units built prior to 1940\n        - DIS      weighted distances to five Boston employment centres\n        - RAD      index of accessibility to radial highways\n        - TAX      full-value property-tax rate per $10,000\n        - PTRATIO  pupil-teacher ratio by town\n        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n        - LSTAT    % lower status of the population\n        - MEDV     Median value of owner-occupied homes in $1000's\n\n    :Missing Attribute Values: None\n\n    :Creator: Harrison, D. and Rubinfeld, D.L.\n\nThis is a copy of UCI ML housing dataset.\nhttp://archive.ics.uci.edu/ml/datasets/Housing\n\n\nThis dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.\n\nThe Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic\nprices and the demand for clean air', J. Environ. Economics & Management,\nvol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics\n...', Wiley, 1980.   N.B. Various transformations are used in the table on\npages 244-261 of the latter.\n\nThe Boston house-price data has been used in many machine learning papers that address regression\nproblems.   \n     \n**References**\n\n   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.\n   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.\n   - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)\n",
 'data': array([[  6.32000000e-03,   1.80000000e+01,   2.31000000e+00, ...,
           1.53000000e+01,   3.96900000e+02,   4.98000000e+00],
        [  2.73100000e-02,   0.00000000e+00,   7.07000000e+00, ...,
           1.78000000e+01,   3.96900000e+02,   9.14000000e+00],
        [  2.72900000e-02,   0.00000000e+00,   7.07000000e+00, ...,
           1.78000000e+01,   3.92830000e+02,   4.03000000e+00],
        ..., 
        [  6.07600000e-02,   0.00000000e+00,   1.19300000e+01, ...,
           2.10000000e+01,   3.96900000e+02,   5.64000000e+00],
        [  1.09590000e-01,   0.00000000e+00,   1.19300000e+01, ...,
           2.10000000e+01,   3.93450000e+02,   6.48000000e+00],
        [  4.74100000e-02,   0.00000000e+00,   1.19300000e+01, ...,
           2.10000000e+01,   3.96900000e+02,   7.88000000e+00]]),
 'feature_names': array(['CRIM', 'ZN', 'INDUS', ..., 'PTRATIO', 'B', 'LSTAT'], 
       dtype='<U7'),
 'target': array([ 24. ,  21.6,  34.7, ...,  23.9,  22. ,  11.9])}

boston.data.shape
Out[19]: (506, 13)

print boston.feature_names
  File "<ipython-input-20-79fe03f8f0a4>", line 1
    print boston.feature_names
               ^
SyntaxError: Missing parentheses in call to 'print'


print( boston.feature_names)
['CRIM' 'ZN' 'INDUS' ..., 'PTRATIO' 'B' 'LSTAT']

print(boston.Desc)
Traceback (most recent call last):

  File "<ipython-input-22-276290dbc45c>", line 1, in <module>
    print(boston.Desc)

  File "C:\Users\deepak\Anaconda3\lib\site-packages\sklearn\datasets\base.py", line 61, in __getattr__
    raise AttributeError(key)

AttributeError: Desc


print(boston.Descr)
Traceback (most recent call last):

  File "<ipython-input-23-64573dce2035>", line 1, in <module>
    print(boston.Descr)

  File "C:\Users\deepak\Anaconda3\lib\site-packages\sklearn\datasets\base.py", line 61, in __getattr__
    raise AttributeError(key)

AttributeError: Descr


print(boston.DESCR)
Boston House Prices dataset
===========================

Notes
------
Data Set Characteristics:  

    :Number of Instances: 506 

    :Number of Attributes: 13 numeric/categorical predictive
    
    :Median Value (attribute 14) is usually the target

    :Attribute Information (in order):
        - CRIM     per capita crime rate by town
        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS    proportion of non-retail business acres per town
        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
        - NOX      nitric oxides concentration (parts per 10 million)
        - RM       average number of rooms per dwelling
        - AGE      proportion of owner-occupied units built prior to 1940
        - DIS      weighted distances to five Boston employment centres
        - RAD      index of accessibility to radial highways
        - TAX      full-value property-tax rate per $10,000
        - PTRATIO  pupil-teacher ratio by town
        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT    % lower status of the population
        - MEDV     Median value of owner-occupied homes in $1000's

    :Missing Attribute Values: None

    :Creator: Harrison, D. and Rubinfeld, D.L.

This is a copy of UCI ML housing dataset.
http://archive.ics.uci.edu/ml/datasets/Housing


This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
prices and the demand for clean air', J. Environ. Economics & Management,
vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
...', Wiley, 1980.   N.B. Various transformations are used in the table on
pages 244-261 of the latter.

The Boston house-price data has been used in many machine learning papers that address regression
problems.   
     
**References**

   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.
   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.
   - many more! (see http://archive.ics.uci.edu/ml/datasets/Housing)


bos = pd.dataframe(boston.data)
Traceback (most recent call last):

  File "<ipython-input-25-e72d214f7ed6>", line 1, in <module>
    bos = pd.dataframe(boston.data)

AttributeError: module 'pandas' has no attribute 'dataframe'


bos = pd.dataFrame(boston.data)
Traceback (most recent call last):

  File "<ipython-input-26-c36d6e58d4a3>", line 1, in <module>
    bos = pd.dataFrame(boston.data)

AttributeError: module 'pandas' has no attribute 'dataFrame'


bos = pd.DataFrame(boston.data)

bos.head()
Out[28]: 
        0     1     2    3      4      5     6       7    8      9     10  \
0  0.00632  18.0  2.31  0.0  0.538  6.575  65.2  4.0900  1.0  296.0  15.3   
1  0.02731   0.0  7.07  0.0  0.469  6.421  78.9  4.9671  2.0  242.0  17.8   
2  0.02729   0.0  7.07  0.0  0.469  7.185  61.1  4.9671  2.0  242.0  17.8   
3  0.03237   0.0  2.18  0.0  0.458  6.998  45.8  6.0622  3.0  222.0  18.7   
4  0.06905   0.0  2.18  0.0  0.458  7.147  54.2  6.0622  3.0  222.0  18.7   

       11    12  
0  396.90  4.98  
1  396.90  9.14  
2  392.83  4.03  
3  394.63  2.94  
4  396.90  5.33  

boston.target[:5]
Out[29]: array([ 24. ,  21.6,  34.7,  33.4,  36.2])

bos.columns = boston.feature_names

bos.head()
Out[31]: 
      CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD    TAX  \
0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900  1.0  296.0   
1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671  2.0  242.0   
2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671  2.0  242.0   
3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622  3.0  222.0   
4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622  3.0  222.0   

   PTRATIO       B  LSTAT  
0     15.3  396.90   4.98  
1     17.8  396.90   9.14  
2     17.8  392.83   4.03  
3     18.7  394.63   2.94  
4     18.7  396.90   5.33  

bos['PRICE'] = boston.target

from sklearn.linear_model import LinearRegression

X = bos.drop('Price', axis = 1)
Traceback (most recent call last):

  File "<ipython-input-34-52029bf81aa5>", line 1, in <module>
    X = bos.drop('Price', axis = 1)

  File "C:\Users\deepak\Anaconda3\lib\site-packages\pandas\core\generic.py", line 2050, in drop
    new_axis = axis.drop(labels, errors=errors)

  File "C:\Users\deepak\Anaconda3\lib\site-packages\pandas\core\indexes\base.py", line 3575, in drop
    labels[mask])

ValueError: labels ['Price'] not contained in axis


X = bos.drop('PRICE', axis = 1)

#This creates a linear regression object

lm = LinearRegression()

l
Traceback (most recent call last):

  File "<ipython-input-38-12f54a96f644>", line 1, in <module>
    l

NameError: name 'l' is not defined


lm
Out[39]: LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

lm.fit(X, bos.PRICE)
Out[40]: LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

print(lm.intercept)
Traceback (most recent call last):

  File "<ipython-input-41-c74cb6f4f6c0>", line 1, in <module>
    print(lm.intercept)

AttributeError: 'LinearRegression' object has no attribute 'intercept'


print(lm.intercept_)
36.4911032804

pre_train = lm.predict(X_train)
Traceback (most recent call last):

  File "<ipython-input-43-26b7e1a7f96f>", line 1, in <module>
    pre_train = lm.predict(X_train)

NameError: name 'X_train' is not defined