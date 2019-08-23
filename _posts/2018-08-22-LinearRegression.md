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

In this post, I will show you how to build a predictive model using the statsmodel api. Before we get into that, let's talk about the data we will be using. There are two datasets that will be used for this predictive model i.e. [County health rankings](https://www.countyhealthrankings.org/) and in particular the *Years of potential life lost (YPPL)* and *Additional measures*

[Years of potential life lost](https://en.wikipedia.org/wiki/Years_of_potential_life_lost) or YPPL is an estimate of the average years a person would have lived if he or she had not died prematurely. It is, therefore, a measure of premature mortality. As an alternative to death rates, it is a method that gives more weight to deaths that occur among younger people. An alternative is to consider the effects of both disability and premature death using disability adjusted life years.

To calculate the years of potential life lost, the analyst has to set an upper reference age. The reference age should correspond roughly to the life expectancy of the population under study. In the developed world, this is commonly set at age 75, but it is essentially arbitrary. Thus, PYLL should be written with respect to the reference age used in the calculation: e.g., YPPL[75].

YPPL can be calculated using individual level data or using age grouped data.[2]

Briefly, for the individual method, each person's YPPL is calculated by subtracting the person's age at death from the reference age. If a person is older than the reference age when he or she dies, that person's YPPL is set to zero (i.e., there are no "negative" YPPLs). In effect, only those who die before the reference age are included in the calculation. Some examples:
1. Reference age = 75; Age at death = 60; PYLL[75] = 75 − 60 = 15
2. Reference age = 75; Age at death = 60; PYLL[75] = 75 − 60 = 15
3. Reference age = 75; Age at death = 80; PYLL[75] = 0 (age at death greater than reference age)

To calculate the YPLL for a particular population in a particular year, the analyst sums the individual PYLLs for all individuals in that population who died in that year. This can be done for all-cause mortality or for cause-specific mortality.