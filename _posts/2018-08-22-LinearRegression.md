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

# turn off future warnings
import warnings
warnings.filterwarnings(action='ignore')


# normalising packages
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
```

```python
ypll = pd.read_csv('ypll.csv') # import data
ypll.head() # view the first 5 rows
```
<!-- image: "/images/linear_reg/regression.jpg" -->
![image-title-here](/images/linear_reg/yppl_head.png){:class="img-responsive"}



```python
# import data analysis libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import ols from sklearn / predictive modeling
from statsmodels.api import OLS

# turn off future warnings
import warnings
warnings.filterwarnings(action='ignore')


# normalising packages
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
```


```python
ypll = pd.read_csv('ypll.csv')
ypll.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FIPS</th>
      <th>State</th>
      <th>County</th>
      <th>Unreliable</th>
      <th>YPLL Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>Alabama</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>10189.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001</td>
      <td>Alabama</td>
      <td>Autauga</td>
      <td>NaN</td>
      <td>9967.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1003</td>
      <td>Alabama</td>
      <td>Baldwin</td>
      <td>NaN</td>
      <td>8322.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1005</td>
      <td>Alabama</td>
      <td>Barbour</td>
      <td>NaN</td>
      <td>9559.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1007</td>
      <td>Alabama</td>
      <td>Bibb</td>
      <td>NaN</td>
      <td>13283.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
add_measures = pd.read_csv('additional_measures_cleaned.csv')
add_measures.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FIPS</th>
      <th>State</th>
      <th>County</th>
      <th>Population</th>
      <th>&lt; 18</th>
      <th>65 and over</th>
      <th>African American</th>
      <th>Female</th>
      <th>Rural</th>
      <th>%Diabetes</th>
      <th>HIV rate</th>
      <th>Physical Inactivity</th>
      <th>mental health provider rate</th>
      <th>median household income</th>
      <th>% high housing costs</th>
      <th>% Free lunch</th>
      <th>% child Illiteracy</th>
      <th>% Drive Alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
      <td>Alabama</td>
      <td>NaN</td>
      <td>4708708</td>
      <td>23.9</td>
      <td>13.8</td>
      <td>26.1</td>
      <td>51.6</td>
      <td>44.6</td>
      <td>12</td>
      <td>NaN</td>
      <td>31</td>
      <td>20</td>
      <td>42586.0</td>
      <td>30</td>
      <td>51.0</td>
      <td>14.8</td>
      <td>84</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001</td>
      <td>Alabama</td>
      <td>Autauga</td>
      <td>50756</td>
      <td>27.8</td>
      <td>11.6</td>
      <td>18.4</td>
      <td>51.4</td>
      <td>44.8</td>
      <td>11</td>
      <td>170.0</td>
      <td>33</td>
      <td>2</td>
      <td>51622.0</td>
      <td>25</td>
      <td>29.0</td>
      <td>12.7</td>
      <td>86</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1003</td>
      <td>Alabama</td>
      <td>Baldwin</td>
      <td>179878</td>
      <td>23.1</td>
      <td>17.0</td>
      <td>10.0</td>
      <td>51.0</td>
      <td>54.2</td>
      <td>10</td>
      <td>176.0</td>
      <td>25</td>
      <td>17</td>
      <td>51957.0</td>
      <td>29</td>
      <td>29.0</td>
      <td>10.6</td>
      <td>83</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1005</td>
      <td>Alabama</td>
      <td>Barbour</td>
      <td>29737</td>
      <td>22.3</td>
      <td>13.8</td>
      <td>46.6</td>
      <td>46.8</td>
      <td>71.5</td>
      <td>14</td>
      <td>331.0</td>
      <td>35</td>
      <td>7</td>
      <td>30896.0</td>
      <td>36</td>
      <td>65.0</td>
      <td>23.2</td>
      <td>82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1007</td>
      <td>Alabama</td>
      <td>Bibb</td>
      <td>21587</td>
      <td>23.3</td>
      <td>13.5</td>
      <td>22.3</td>
      <td>48.0</td>
      <td>81.5</td>
      <td>11</td>
      <td>90.0</td>
      <td>37</td>
      <td>0</td>
      <td>41076.0</td>
      <td>18</td>
      <td>48.0</td>
      <td>17.5</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>




```python
ypll.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3192 entries, 0 to 3191
    Data columns (total 5 columns):
    FIPS          3192 non-null int64
    State         3192 non-null object
    County        3141 non-null object
    Unreliable    196 non-null object
    YPLL Rate     3097 non-null float64
    dtypes: float64(1), int64(1), object(3)
    memory usage: 124.8+ KB



```python
ypll.drop(index=ypll[ypll.Unreliable.isin(['x'])].index, inplace=True)
```


```python
add_measures.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3192 entries, 0 to 3191
    Data columns (total 18 columns):
    FIPS                           3192 non-null int64
    State                          3192 non-null object
    County                         3141 non-null object
    Population                     3192 non-null int64
    < 18                           3192 non-null float64
    65 and over                    3192 non-null float64
    African American               3192 non-null float64
    Female                         3192 non-null float64
    Rural                          3191 non-null float64
    %Diabetes                      3192 non-null int64
    HIV rate                       2230 non-null float64
    Physical Inactivity            3192 non-null int64
    mental health provider rate    3192 non-null int64
    median household income        3191 non-null float64
    % high housing costs           3192 non-null int64
    % Free lunch                   3173 non-null float64
    % child Illiteracy             3186 non-null float64
    % Drive Alone                  3192 non-null int64
    dtypes: float64(9), int64(7), object(2)
    memory usage: 449.0+ KB



```python
add_measures.drop(index=add_measures[add_measures['% child Illiteracy'].isnull()].index, inplace=True)
```


```python
add_measures.drop(index=add_measures[add_measures['County'].isnull()].index, inplace=True)
```


```python
ypll.columns
```




    Index(['FIPS', 'State', 'County', 'Unreliable', 'YPLL Rate'], dtype='object')




```python
cols = [
    'FIPS',
#  'State',
 'County',
 'Population',
 '< 18',
 '65 and over',
 'African American',
 'Female',
 'Rural',
 '%Diabetes',
 'HIV rate',
 'Physical Inactivity',
 'mental health provider rate',
 'median household income',
 '% high housing costs',
 '% Free lunch',
 '% child Illiteracy',
 '% Drive Alone'
]

left = add_measures[cols]
left.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FIPS</th>
      <th>County</th>
      <th>Population</th>
      <th>&lt; 18</th>
      <th>65 and over</th>
      <th>African American</th>
      <th>Female</th>
      <th>Rural</th>
      <th>%Diabetes</th>
      <th>HIV rate</th>
      <th>Physical Inactivity</th>
      <th>mental health provider rate</th>
      <th>median household income</th>
      <th>% high housing costs</th>
      <th>% Free lunch</th>
      <th>% child Illiteracy</th>
      <th>% Drive Alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1001</td>
      <td>Autauga</td>
      <td>50756</td>
      <td>27.8</td>
      <td>11.6</td>
      <td>18.4</td>
      <td>51.4</td>
      <td>44.8</td>
      <td>11</td>
      <td>170.0</td>
      <td>33</td>
      <td>2</td>
      <td>51622.0</td>
      <td>25</td>
      <td>29.0</td>
      <td>12.7</td>
      <td>86</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1003</td>
      <td>Baldwin</td>
      <td>179878</td>
      <td>23.1</td>
      <td>17.0</td>
      <td>10.0</td>
      <td>51.0</td>
      <td>54.2</td>
      <td>10</td>
      <td>176.0</td>
      <td>25</td>
      <td>17</td>
      <td>51957.0</td>
      <td>29</td>
      <td>29.0</td>
      <td>10.6</td>
      <td>83</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1005</td>
      <td>Barbour</td>
      <td>29737</td>
      <td>22.3</td>
      <td>13.8</td>
      <td>46.6</td>
      <td>46.8</td>
      <td>71.5</td>
      <td>14</td>
      <td>331.0</td>
      <td>35</td>
      <td>7</td>
      <td>30896.0</td>
      <td>36</td>
      <td>65.0</td>
      <td>23.2</td>
      <td>82</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1007</td>
      <td>Bibb</td>
      <td>21587</td>
      <td>23.3</td>
      <td>13.5</td>
      <td>22.3</td>
      <td>48.0</td>
      <td>81.5</td>
      <td>11</td>
      <td>90.0</td>
      <td>37</td>
      <td>0</td>
      <td>41076.0</td>
      <td>18</td>
      <td>48.0</td>
      <td>17.5</td>
      <td>83</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1009</td>
      <td>Blount</td>
      <td>58345</td>
      <td>24.2</td>
      <td>14.7</td>
      <td>2.1</td>
      <td>50.2</td>
      <td>91.0</td>
      <td>11</td>
      <td>66.0</td>
      <td>35</td>
      <td>2</td>
      <td>46086.0</td>
      <td>21</td>
      <td>37.0</td>
      <td>13.9</td>
      <td>80</td>
    </tr>
  </tbody>
</table>
</div>




```python
ypll.drop(index=ypll[ypll.County.isnull()].index, inplace=True)
```


```python
len(ypll)
```




    2945




```python
df = pd.merge(left, ypll, on=['County', 'FIPS'] )
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FIPS</th>
      <th>County</th>
      <th>Population</th>
      <th>&lt; 18</th>
      <th>65 and over</th>
      <th>African American</th>
      <th>Female</th>
      <th>Rural</th>
      <th>%Diabetes</th>
      <th>HIV rate</th>
      <th>Physical Inactivity</th>
      <th>mental health provider rate</th>
      <th>median household income</th>
      <th>% high housing costs</th>
      <th>% Free lunch</th>
      <th>% child Illiteracy</th>
      <th>% Drive Alone</th>
      <th>State</th>
      <th>Unreliable</th>
      <th>YPLL Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>Autauga</td>
      <td>50756</td>
      <td>27.8</td>
      <td>11.6</td>
      <td>18.4</td>
      <td>51.4</td>
      <td>44.8</td>
      <td>11</td>
      <td>170.0</td>
      <td>33</td>
      <td>2</td>
      <td>51622.0</td>
      <td>25</td>
      <td>29.0</td>
      <td>12.7</td>
      <td>86</td>
      <td>Alabama</td>
      <td>NaN</td>
      <td>9967.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1003</td>
      <td>Baldwin</td>
      <td>179878</td>
      <td>23.1</td>
      <td>17.0</td>
      <td>10.0</td>
      <td>51.0</td>
      <td>54.2</td>
      <td>10</td>
      <td>176.0</td>
      <td>25</td>
      <td>17</td>
      <td>51957.0</td>
      <td>29</td>
      <td>29.0</td>
      <td>10.6</td>
      <td>83</td>
      <td>Alabama</td>
      <td>NaN</td>
      <td>8322.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1005</td>
      <td>Barbour</td>
      <td>29737</td>
      <td>22.3</td>
      <td>13.8</td>
      <td>46.6</td>
      <td>46.8</td>
      <td>71.5</td>
      <td>14</td>
      <td>331.0</td>
      <td>35</td>
      <td>7</td>
      <td>30896.0</td>
      <td>36</td>
      <td>65.0</td>
      <td>23.2</td>
      <td>82</td>
      <td>Alabama</td>
      <td>NaN</td>
      <td>9559.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1007</td>
      <td>Bibb</td>
      <td>21587</td>
      <td>23.3</td>
      <td>13.5</td>
      <td>22.3</td>
      <td>48.0</td>
      <td>81.5</td>
      <td>11</td>
      <td>90.0</td>
      <td>37</td>
      <td>0</td>
      <td>41076.0</td>
      <td>18</td>
      <td>48.0</td>
      <td>17.5</td>
      <td>83</td>
      <td>Alabama</td>
      <td>NaN</td>
      <td>13283.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1009</td>
      <td>Blount</td>
      <td>58345</td>
      <td>24.2</td>
      <td>14.7</td>
      <td>2.1</td>
      <td>50.2</td>
      <td>91.0</td>
      <td>11</td>
      <td>66.0</td>
      <td>35</td>
      <td>2</td>
      <td>46086.0</td>
      <td>21</td>
      <td>37.0</td>
      <td>13.9</td>
      <td>80</td>
      <td>Alabama</td>
      <td>NaN</td>
      <td>8475.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['YPLL Rate'].fillna(value=df['YPLL Rate'].mean(), inplace=True)
df.drop(labels=['Unreliable'], inplace=True, axis=1)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2941 entries, 0 to 2940
    Data columns (total 19 columns):
    FIPS                           2941 non-null int64
    County                         2941 non-null object
    Population                     2941 non-null int64
    < 18                           2941 non-null float64
    65 and over                    2941 non-null float64
    African American               2941 non-null float64
    Female                         2941 non-null float64
    Rural                          2941 non-null float64
    %Diabetes                      2941 non-null int64
    HIV rate                       2220 non-null float64
    Physical Inactivity            2941 non-null int64
    mental health provider rate    2941 non-null int64
    median household income        2941 non-null float64
    % high housing costs           2941 non-null int64
    % Free lunch                   2926 non-null float64
    % child Illiteracy             2941 non-null float64
    % Drive Alone                  2941 non-null int64
    State                          2941 non-null object
    YPLL Rate                      2941 non-null float64
    dtypes: float64(10), int64(7), object(2)
    memory usage: 459.5+ KB



```python
df['% Free lunch'].fillna(value=df['% Free lunch'].mean(), inplace=True)
df['HIV rate'].fillna(value=df['HIV rate'].mean(), inplace=True)
```

## Visualising correlations


```python
import seaborn as sns
sns.set()
```


```python
fig = plt.figure(figsize=(10,15))
plt.tight_layout()
fig.suptitle('Visualising correlations', y=0.9, fontweight=1000)
subplot = fig.add_subplot(3,1,1)

subplot.scatter(df['YPLL Rate'], df['%Diabetes'], color='orange')
subplot.set_xlabel('ypll rate')
subplot.set_ylabel('% diabetes')

subplot = fig.add_subplot(3,1,2)
subplot.scatter(df['YPLL Rate'], df['< 18'], color='red')
subplot.set_xlabel('ypll rate')
subplot.set_ylabel('% < 18')


subplot = fig.add_subplot(3,1,3)
subplot.scatter(df['YPLL Rate'], df['median household income'], color='green')
subplot.set_xlabel('ypll rate')
subplot.set_ylabel('median household income');
```


![png](output_18_0.png)


### visualising other variables vs ypll


```python
plot=[
 'FIPS',
 'Population',
#  '< 18',
 '65 and over',
 'African American',
 'Female',
 'Rural',
#  '%Diabetes',
 'HIV rate',
 'Physical Inactivity',
 'mental health provider rate',
#  'median household income',
 '% high housing costs',
 '% Free lunch',
 '% child Illiteracy',
 '% Drive Alone',
 'YPLL Rate'
]
```


```python
fig = plt.figure(figsize=(15,35))
# plt.tight_layout()
fig.suptitle('Visualising correlations', y=0.9, fontweight=1000)

for i in range(len(plot)): # plot different attributes
    
    subplot = fig.add_subplot(len(plot),3,i+1)

    subplot.scatter(df['YPLL Rate'], df[plot[i]], color='green')
    subplot.set_xlabel('ypll rate')
    subplot.set_ylabel(plot[i])
```


![png](output_21_0.png)



```python
# correlation matrix
# sort by the top 8 values
cor_matrx=df.corr()
cor_matrx['YPLL Rate'].sort_values(ascending=False)
```




    YPLL Rate                      1.000000
    % Free lunch                   0.694590
    %Diabetes                      0.662009
    Physical Inactivity            0.645538
    % child Illiteracy             0.461458
    African American               0.436223
    Rural                          0.286216
    HIV rate                       0.175228
    < 18                           0.127011
    Female                         0.116916
    65 and over                    0.071458
    % Drive Alone                  0.031896
    FIPS                          -0.062884
    % high housing costs          -0.106475
    Population                    -0.162204
    mental health provider rate   -0.177113
    median household income       -0.634813
    Name: YPLL Rate, dtype: float64



# YPLL vs. % Diabetes regression



```python
# model = OLS # rename model

# define independent and dependent variable
X = df[['%Diabetes']]
y = df['YPLL Rate']
```


```python
# fit model and predict using OLS (same as LinearRegression)
# show summary
# add bias term 

import statsmodels
model = OLS(y, statsmodels.tools.add_constant(X)).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>YPLL Rate</td>    <th>  R-squared:         </th> <td>   0.438</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.438</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2293.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 21 Aug 2019</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>12:50:10</td>     <th>  Log-Likelihood:    </th> <td> -26260.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2941</td>      <th>  AIC:               </th> <td>5.252e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2939</td>      <th>  BIC:               </th> <td>5.254e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>     <td>  809.4571</td> <td>  163.462</td> <td>    4.952</td> <td> 0.000</td> <td>  488.946</td> <td> 1129.969</td>
</tr>
<tr>
  <th>%Diabetes</th> <td>  767.1553</td> <td>   16.021</td> <td>   47.884</td> <td> 0.000</td> <td>  735.742</td> <td>  798.569</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1081.131</td> <th>  Durbin-Watson:     </th> <td>   1.545</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>7964.973</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 1.552</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>       <td>10.441</td>  <th>  Cond. No.          </th> <td>    50.0</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



### Intepreting the math

- let's intepret the data and check the statistical significance to make sure nothing happend by chance and that our regression is meaningful !

- Prob(F-statistic) is close to 0 which is less than 0.05 or 0.01, so we have significance & thus we can safely intepret the rest of the data !

- for this model the line of best fit is represented by y = mx + constant or ypll = 767.15 * $\%$ diabetes + 809.45

- To check how well this line/model performs, the R- squared value of 0.438 is recorded. This means that 43.8$\%$ of the changes in y (YPLL rate) can be predicted by changes in x ($\%$Diabetes)


###  what does this all mean?

- Simply put, it means that without knowing the YPLL of a community, we can simply collect data about percentage of people affected by diabetes and construct 43.8$\%$ of the YPLL rate behaviour

### visualize the model built above



```python
# print cofficients
print(model.params.index)
model.params.values
```

    Index(['const', '%Diabetes'], dtype='object')





    array([809.45713922, 767.15527376])




```python
fig = plt.figure(figsize=(10,8))
fig.suptitle('Plot Showing how model fits data', fontweight=1000, y=0.92)
subplot =fig.add_subplot(111)
subplot.scatter(X,y, color='darkblue', label='scatter plot')
subplot.set_xlabel('%Diabetes')
subplot.set_ylabel('YPLL Rate')

# y = mx + c
Y = model.params.values[1] * X['%Diabetes'] + model.params.values[0]
subplot.plot(X['%Diabetes'], Y, color='darkorange', label='models best fit')
subplot.legend(ncol=3);

```


![png](output_32_0.png)



## Running the correlations for the < 18 and median household income

#### YPLL vs.< 18


```python
# Define new X and y 
X = df['< 18']
y = df['YPLL Rate']
```


```python
# fit and show summary
model = OLS(y, statsmodels.tools.add_constant(X)).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>YPLL Rate</td>    <th>  R-squared:         </th> <td>   0.016</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.016</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   48.19</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 21 Aug 2019</td> <th>  Prob (F-statistic):</th> <td>4.74e-12</td> 
</tr>
<tr>
  <th>Time:</th>                 <td>12:50:12</td>     <th>  Log-Likelihood:    </th> <td> -27084.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2941</td>      <th>  AIC:               </th> <td>5.417e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2939</td>      <th>  BIC:               </th> <td>5.418e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td> 6302.8803</td> <td>  315.173</td> <td>   19.998</td> <td> 0.000</td> <td> 5684.899</td> <td> 6920.862</td>
</tr>
<tr>
  <th>< 18</th>  <td>   91.8513</td> <td>   13.232</td> <td>    6.942</td> <td> 0.000</td> <td>   65.907</td> <td>  117.796</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>407.946</td> <th>  Durbin-Watson:     </th> <td>   1.290</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 824.396</td> 
</tr>
<tr>
  <th>Skew:</th>          <td> 0.848</td>  <th>  Prob(JB):          </th> <td>9.65e-180</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.962</td>  <th>  Cond. No.          </th> <td>    169.</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



- Again, let's do some sanity checks to make sure results are not random
- Prob(F-statistic) close to 0 which is less than 0.05 or 0.01 , so we have significance
- R squared value not so strong. Only about 1 percent of the changes in y can be explained by the changes in x

### Visualising model on this data


```python
fig = plt.figure(figsize=(10,8))
fig.suptitle('Plot Showing how model fits data', fontweight=1000, y=0.92)
subplot =fig.add_subplot(111)
subplot.scatter(X,y, color='darkorange', label='scatter plot')
subplot.set_xlabel('< 18')
subplot.set_ylabel('YPLL Rate')

# y = mx + c
Y = model.params.values[1] * X + model.params.values[0]
subplot.plot(X, Y, color='black', label='models best fit')
subplot.legend(ncol=4);

```


![png](output_39_0.png)


#### YPLL vs.median household income


```python
# Define new X and y 
X = df['median household income']
y = df['YPLL Rate']
```


```python
model = OLS(y, statsmodels.tools.add_constant(X)).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>YPLL Rate</td>    <th>  R-squared:         </th> <td>   0.403</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.403</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   1984.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 21 Aug 2019</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>12:50:14</td>     <th>  Log-Likelihood:    </th> <td> -26349.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2941</td>      <th>  AIC:               </th> <td>5.270e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2939</td>      <th>  BIC:               </th> <td>5.271e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
             <td></td>                <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                   <td> 1.438e+04</td> <td>  137.216</td> <td>  104.809</td> <td> 0.000</td> <td> 1.41e+04</td> <td> 1.47e+04</td>
</tr>
<tr>
  <th>median household income</th> <td>   -0.1331</td> <td>    0.003</td> <td>  -44.540</td> <td> 0.000</td> <td>   -0.139</td> <td>   -0.127</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>697.404</td> <th>  Durbin-Watson:     </th> <td>   1.377</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>3028.219</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.085</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.472</td>  <th>  Cond. No.          </th> <td>1.81e+05</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.81e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



- Prob(F-statistic) close to 0 which is less than 0.05 or 0.01 , so we have significance
- R -squared value is 0.403. This means that 40$\%$ of the changes in YPLL Rate can be explained by the changes in dependent variable x i.e median household income
- median household income is also negatively correlated

### what is the meaning of this?
- We can constract 40$\%$ of the information about YPLL Rate by collecting data about median household income.
- Percentage of those below 18 i.e [% < 18] is a weak predictor too, almost no changes are explained by it in x

### Visualize this model


```python
fig = plt.figure(figsize=(10,8))
fig.suptitle('Plot Showing how model fits data', fontweight=1000, y=0.92)
subplot =fig.add_subplot(111)
subplot.scatter(X,y, color='darkgreen', label='scatter plot')
subplot.set_xlabel('median household income')
subplot.set_ylabel('YPLL Rate')

# y = mx + c
Y = model.params.values[1] * X + model.params.values[0]
subplot.plot(X, Y, color='black', label='models best fit')
subplot.legend(ncol=4);
```


![png]("/images/linear_reg/output_46_0.png")


### Multiple Linear regression

- combining infomation from multiple measures to improve model


```python
# define X and y
cols_to_use =[
#     'FIPS',
#  'Population',
 '<_18',
#  '65_and_over',
#  'African_American',
#  'Female',
#  'Rural',
 '%Diabetes',
 'HIV_rate',
 'Physical_Inactivity',
 'mental_health_provider_rate',
 'median_household_income',
#  '%_high_housing_costs',
 '%_Free_lunch',
 '%_child_Illiteracy',
 '%_Drive_Alone',
#  'YPLL_Rate'
]

X = df[cols_to_use]
X_scaled = scaler.fit_transform(X)



y = df['YPLL_Rate']
```


```python
model = OLS(y, statsmodels.tools.add_constant(X)).fit()
model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>YPLL_Rate</td>    <th>  R-squared:         </th> <td>   0.645</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.644</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   592.0</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Wed, 21 Aug 2019</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>15:17:05</td>     <th>  Log-Likelihood:    </th> <td> -25584.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2941</td>      <th>  AIC:               </th> <td>5.119e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2931</td>      <th>  BIC:               </th> <td>5.125e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     9</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
               <td></td>                  <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>                       <td> 4156.5198</td> <td>  400.848</td> <td>   10.369</td> <td> 0.000</td> <td> 3370.548</td> <td> 4942.491</td>
</tr>
<tr>
  <th><_18</th>                        <td>   87.2481</td> <td>    9.070</td> <td>    9.620</td> <td> 0.000</td> <td>   69.464</td> <td>  105.032</td>
</tr>
<tr>
  <th>%Diabetes</th>                   <td>  328.7789</td> <td>   22.331</td> <td>   14.723</td> <td> 0.000</td> <td>  284.992</td> <td>  372.566</td>
</tr>
<tr>
  <th>HIV_rate</th>                    <td>    0.9893</td> <td>    0.149</td> <td>    6.656</td> <td> 0.000</td> <td>    0.698</td> <td>    1.281</td>
</tr>
<tr>
  <th>Physical_Inactivity</th>         <td>   80.4428</td> <td>    8.722</td> <td>    9.223</td> <td> 0.000</td> <td>   63.342</td> <td>   97.544</td>
</tr>
<tr>
  <th>mental_health_provider_rate</th> <td>   -1.1311</td> <td>    0.448</td> <td>   -2.527</td> <td> 0.012</td> <td>   -2.009</td> <td>   -0.253</td>
</tr>
<tr>
  <th>median_household_income</th>     <td>   -0.0481</td> <td>    0.004</td> <td>  -13.447</td> <td> 0.000</td> <td>   -0.055</td> <td>   -0.041</td>
</tr>
<tr>
  <th>%_Free_lunch</th>                <td>   47.3245</td> <td>    2.820</td> <td>   16.779</td> <td> 0.000</td> <td>   41.794</td> <td>   52.855</td>
</tr>
<tr>
  <th>%_child_Illiteracy</th>          <td>  -43.7611</td> <td>    6.407</td> <td>   -6.830</td> <td> 0.000</td> <td>  -56.324</td> <td>  -31.198</td>
</tr>
<tr>
  <th>%_Drive_Alone</th>               <td>  -31.4760</td> <td>    4.133</td> <td>   -7.616</td> <td> 0.000</td> <td>  -39.580</td> <td>  -23.372</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>779.012</td> <th>  Durbin-Watson:     </th> <td>   1.748</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>4556.589</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 1.125</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 8.668</td>  <th>  Cond. No.          </th> <td>6.86e+05</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 6.86e+05. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



- Analysis
----
- free lunch is strongly correlated to YPLL rate
- should we eliminate free lunch to improve life and reduce YPLL Rate?
- or peharps there is a third variable connecting everything that policy makers should look into?

## Looking at Non-linearity..

- linear regression only helps us come up with the best fit line for our data
- But there are cases of weak interaction between variables (non-linearity)
- let's try fitting with a polynomial fit using sklearn


```python
from sklearn.preprocessing import PolynomialFeatures
```


```python
poly_transform = PolynomialFeatures(degree=3)

X_ply = poly_transform.fit_transform(X)
model = OLS(y, statsmodels.tools.add_constant(X_ply)).fit()
```


```python
print('R-squared value: {:.2f}'.format(model.rsquared))
print('Adjusted R-squared value: {:.2f}'.format(model.rsquared_adj))
```

    R-squared value: 0.76
    Adjusted R-squared value: 0.74


conclusion
--
- The initial Adj R squared value of .64 has been improved upon greatly using the polynomial features method.
- The new R squared values are 0.76 and 0.74 for the adjusted!
- Note: Train test split procedure was not used
- To be done in a future notebook using the same data with metrics!
- This was an illustration of how to back a model from a statistical point of view..