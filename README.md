# Machine Powered Investing in Lending Club
**Description:**
Machine Learning algorithms were used to create a loan classification model for conservative investors using data from the Lending Club website. Prior to developing the model, the data obtained was cleaned and explored. The work was completed in three notebooks:
* [Data Wrangling Notebook](https://github.com/paulb17/Machine-Powered-Investing-in-Lending-Club/blob/master/Data_Wrangling%20.ipynb)
* [Data Exploration Notebook](https://github.com/paulb17/Machine-Powered-Investing-in-Lending-Club/blob/master/Data_Exploration.ipynb)
* [Machine Learning Notebook](https://github.com/paulb17/Machine-Powered-Investing-in-Lending-Club/blob/master/Data_Modeling.ipynb)

**Key Skills:**
* Data Wrangling using Pandas
* Data Visualization using seaborn and matplotlib
* Inferential Statistics using scipy
    * Hypothesis testing: t-test, Chi-square test and Spearman's rank correlation test
* Machine Learning using sickit-learn
    * Algorithms: Logistic Regression, Kernel SVM (SVC), Random Forest Classifier, XGBoost, Voting Classifier and K-means clustering

## [Data Wrangling Notebook](https://github.com/paulb17/Machine-Powered-Investing-in-Lending-Club/blob/master/Data_Wrangling%20.ipynb)
In this notebook raw Lending Club data was cleaned to remove information an investor would not have at the time of making the investment. A brief summary of the steps taken is listed below:

**Removing Extraneous Data**
1. Removing columns with 100% missing values.
2. Removing columns based on description that: 
    * Leaked information from the future.
    * Contained redundant information.
3. Removing columns with only one unique value.  

**Preparing features for data exploration and machine learning**
1. Preparing Categorical columns by:
    * Mapping ordinal values to integers.
    * Encoding nominal values as dummy variables.
2. Removing percentage signs from continous data. 
3. Preparing the target column.
4. Handling missing values by:
    * Dropping rows with missing values under certain criteria.
    * Imputing missing values using observations from data.
    
## [Data Exploration Notebook](https://github.com/paulb17/Machine-Powered-Investing-in-Lending-Club/blob/master/Data_Exploration.ipynb)
In this notebook a few important metrics for investing were investigated using inferential and descriptive statistics. A 5% significance level was used for all statistical tests. 

### Fiar Isaac Corporation (FICO) Scores
![title](https://github.com/paulb17/Machine-Powered-Investing-in-Lending-Club/blob/master/README_images%20/FICO%20Score%20Analysis.png)

Results of a t-test indictated that on average, borrowers that default have lower FICO scores than those that fully pay off their loans (p = .00027). In addition, a Spearman's rank correlation test led to the conclusion that borrowers with higher FICO scores are less likely to default (&rho; = -1.0, p <.0001). Based on ratings from the FICO website, Lending Club borrowers have FICO scores that can be categorized as good, very good and exceptional. 

### Age of Oldest Credit Account 
![title](https://github.com/paulb17/Machine-Powered-Investing-in-Lending-Club/blob/master/README_images%20/Credit_history.png)

Although a positive association was found between FICO scores and the age of a borrower's oldest credit account (&rho; = .92, p = .00086), it was noted that default risk did not generally decrease as the age of oldest credit account increased. This was mainly as a result of the significantly larger proportion of borrowers in the 500-600 months oldest credit account category that loaned money for home improvement (p = .002). 

Further investigation revealed that most borrowers who loan money for home imporovements have mortgages. However, a chi-square test led to the conclusion that borrowers with mortgages are less likely to default than borrowers without mortgages (p < .0001).  

### Employment History, Annual Income and Loan Amount
![title](https://github.com/paulb17/Machine-Powered-Investing-in-Lending-Club/blob/master/README_images%20/Inc_emp_loan.png)

FICO score models do not make use of employment or annual income in computing a borrowers credit score as they have found it is not a good predictor of future credit performance. In line with the findings of FICO, Spearman's rank correlation test indicated that there's no association between the employment length and default rate of borrowers in the Lending Club marketplace (p = .81).  

However, with the exception of annual incomes in the $350,000 - $400,000, it was found that the percentage of borrowers that defaulted decreased as annual income increased. The unexpected deviation in default rate was potentially as a result of the small count of borrowers in the $350,000 - $400,000 income category. Further investigation using a t-test  revealed that the default rate in the $350,000 - $400,000 income category is not significantly different than its two neighbouring income categories, $300,000 - $350,000 and $400,000 - $450,000 (p = .16). 

Finally, it was found that as the loan amount requested increased, the default rate also increased. 


### Usefulness of categorized employer data and subgrade data
![title](https://github.com/paulb17/Machine-Powered-Investing-in-Lending-Club/blob/master/README_images%20/emp_grade.png)

In the uncleaned Lending Club Dataset borrowers provided their employer title. The responses borrowers provided were not selected from a categorized list, and would therefore take a significant amount of time to clean. In order to  evaluate the usefulness of categorizing the employer titles without spending too much time, approximately 15% of the responses provided were categorized. For each category, a t-test was performed comparing the default rate of borrowers within the category to all borrowers outside the category. Using the Bonferroni correction to account for the multiple testing, it was found that 2 of the 8 employer categories have significantly different default rates relative to borrowers outside the category. These categories included borrowers that provided no responses and borrowers that work in educationalor research institutions. Given this result, it will likely be beneficial to fully categorize the employer data as it could provide useful insights to investors.

For grades A, B, C and D, the default risk generally increased as the sub-grades increased. For grades D, E and F there appeared to be some randomness in the variation of default risk with sub-grade. Spearman's rank correlation test suggests that there is a positive association between sub grades and default rate of borrowers (&rho; = .96, p <.0001). Accordingly, it was concluded that the sub-grade column offers useful insights to investors. 

## [Data Modeling Notebook](https://github.com/paulb17/Machine-Powered-Investing-in-Lending-Club/blob/master/Data_Modeling.ipynb)
In this notebook, a variety of machine learning algorithms were used to try and cluster and model the data.

### Clustering 
![title](https://github.com/paulb17/Machine-Powered-Investing-in-Lending-Club/blob/master/README_images%20/clustering.png)

K-Means clustering was used to investigate whether customer segments existed. To determine the optimal number of clusters I attempted using the elbow method; however, no clear elbow was observed. Accordingly, the silhouette method was used and the optimal number of clusters was found to be 2. Further investigation into the features of both clusters revealed that there is no substantial difference worth investigating.

### Parameter Tuning and Model Comparison
![title](https://github.com/paulb17/Machine-Powered-Investing-in-Lending-Club/blob/master/README_images%20/model_tuning.png)

The Lending Club data was divided into a training set (75% the size of the wrangled data) and a hold-out set (25% the size of the wrangled data). The logistic regression, random forest and kernel SVM classifiers were tuned via grid or random searches with 10 fold cross validation. For each model, the parameters that were tuned are listed below:

* Logistic Regression: C and class_weight
* Random Forest Classifier: class_weight, max_features, min_samples_split, min_samples_leaf, n_estimators, bootstrap and criterion
* Kernel SVM (SVC): class_weight, gamma and C

Given the imbalanced nature of the loan data, area under the receiver operating characteristic (AUROC) appeared to be the appropriate score to use to compare each model. Upon completion of tuning, the logistic regression model was found to have the best AUROC score of 0.696

### PCA on Logistic Regression Model
![title](https://github.com/paulb17/Machine-Powered-Investing-in-Lending-Club/blob/master/README_images%20/PCA.png)

As an attempt to improve the AUROC of the hold-out set, the logistic regression model was implemented while varying the number of principal components used. The following was found during the investigation: 
* 71 of the 78 components account for a 100% of the explained variance in the features
* The AUROC of the hold-out set appears to increase as the number of principal components increases with the exception of a dip between 30 and 50 principal components. 

### Boosting and Blending Ensemble methods
![title](https://github.com/paulb17/Machine-Powered-Investing-in-Lending-Club/blob/master/README_images%20/ensemble.png)

To try and improve the AUROC of the hold-out set, an XGBoost and a voting classifier were tested. The voting classifier was composed of an XGBoost classifier and the optimal logistic regression model. The XGBoost classifier had a hold-out set AUROC of 0.706, while the voting classifier had a hold-out set AUROC of 0.705. It was noted that the XGBoost classifier was better at identifying true postives (fully paid loans) while the voting classifier was better at identifying true negatives (charged off loans). 


### Conclusion
![title](https://github.com/paulb17/Machine-Powered-Investing-in-Lending-Club/blob/master/README_images%20/conclusion.png)

The primary goal of this project was to develop loan classification models for the conservative investor. To illustrate how to use the models, it wsa assumed the conservative investor is only willing to accept a default risk of ~5%. To this end, the penalty parameters of the three best models were tweaked till the predictions of fully paid loans in the hold-out set had a default risk less than 5%.  

A review of the results reveals that the logistic regression model has the fewest number of false and true positives while the XGB model has the largest number of false and true positives. 

Tweaking the penalty parameters of each model can further reduce the number of false positives; however, this will be at the expense of the number of true positives. For instance, let's consider the test data which is composed of 9,930 borrowers. We know that 8539 of these borrowers fully paid their loans. However, the XGB boost model with a default risks of ~5% only correctly identifies 2742 of these borrowers. The previous XGBoost model which had a default risk of ~8% was able to identify 5232 of these borrowers. 

In short, while it is possible to develop models with very low default risks, investors will have to be willing to patiently search for borrowers that meet their criteria. According to the [blog post](https://www.moneyunder30.com/lending-club-investing) of David Weliver, a long term Lending Club investor and founder of Money Under 30, this may be difficult because institutional investors are using algorithms and analysts to find the best loans. Consequently, the less risky loans are funded very quickly.  

Finally, it is worth noting that tweaking the model parameters so that the default risk in the hold-out set is 0 does not guarantee that a borrower invested in will not default. Models developed do not account for all forms of risks (e.g. people losing all their property to a wildfire).   

Further work can be done to try and improve the models developed or advance this work. A few considerations include:
* Testing other algorithms (e.g neural network) and other ensemble methods (e.g stack ensemble). 
* Including the more recent Lending Club data available on the Lending Club Website.
* Fully categorizing the employer data.
* Performing thorough grid searches on different SVM kernels and the XGBoost model.


