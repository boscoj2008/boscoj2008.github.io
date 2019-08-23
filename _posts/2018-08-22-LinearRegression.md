---
title: "Machine Learning: Ordinary least squares using statsmodel api"
date: 2018-08-22
tags: [machine learning, data science, regression]
header:
  image: "/images/linear_reg/regression.jpg"
excerpt: "Machine Learning, Linear Regression, Polynomial Regression, Data Science"
---

# Regression: Building a predictive model

Linear regression or the ordinary least sqaures method, is a supervised machine learning technique that is used to predict a continuos valued output. Everytime we fit a multiple linear regression model to our data, we compute a set of weights or coefficients, $\beta_0,\beta_1,\beta_2, \beta_3...\beta_n$ where $\beta_0$ is the intercept or the constant plus a bias term also called the error term $\epsilon$.

In this post, I will show you how to build a predictive model using the statsmodel api. Before we get into that, let's talk about the data we will be using. There are two datasets that will be used for this predictive model i.e. [County health rankings](https://www.countyhealthrankings.org/) and in particular the *[Years of potential life lost (YPPL)](https://www.countyhealthrankings.org/explore-health-rankings/rankings-data-documentation/national-data-documentation-2010-2017)* and *Additional measures data* of which you can find the cleaned up versions on my github.

[Years of potential life lost](https://en.wikipedia.org/wiki/Years_of_potential_life_lost) or YPPL is an estimate of the average years a person would have lived if he or she had not died prematurely. It is, therefore, a measure of premature mortality. As an alternative to death rates, it is a method that gives more weight to deaths that occur among younger people.

To calculate the years of potential life lost or YPPL, an upper reference age is determined. The reference age should correspond roughly to the life expectancy of the population under study. In this data, the reference age is 75. So if a person dies at 65, their YPPL is calculated as: 75 - 65 = 10 and so on.

Now, let's get into the data. First step of any data science project is to clean the data. This may include handling missing values if any, normalizing our data, perfoming One Hot encoding for categorical variables etc. Machine learning models do not work if our data have missing values so it's very important that we check for missing values.

python code block to import data analysis packages:
```python
# import data analysis libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import ols from sklearn / predictive modeling
from statsmodels.api import OLS
import pandas_profiling as pp

# turn off future warnings
import warnings
warnings.filterwarnings(action='ignore')


# normalising packages
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
```
