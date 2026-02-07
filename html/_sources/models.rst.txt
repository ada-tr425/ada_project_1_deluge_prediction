====================================
Regression and Classification models
====================================

Risk Class Prediction Models
============================

All Zero Risk
-------------

This model assumes all unlabeled flood risk data is near zero risk (band 1), which is the modal class for the data set. Since the dataset is strongly unbalanced, this is significantly more probable for a random class than any other band.

This is a baseline model to compare other more skillful models against. While it achieve a moderately good accuracy, it is not a useful model for risk prediction, and performs poorly on most other metrics, including the scoring metric specified in the instructions.

Seven Class Flood Probability
-----------------------------
A Random Forest classifier was employed to predict seven-class flood risk, using a unified preprocessing pipeline and KNN-based spatial matching. 
The model was evaluated against simpler baselines such as Logistic Regression and KNN classification, which underperformed on accuracy and non-linear patterns. 
Elevation, soil type and distance to watercourses emerged as the most informative features, while postcode-sector metadata and historic flooding contributed little due to sparsity.

Historic Flooding Classification
-------------------------------
A Random Forest classifier was developed for historic-flooding prediction and evaluated against a Decision Tree baseline.
Using postcode-level labelled data with unified preprocessing, stratified 5-fold cross-validation showed that the Random Forest achieved higher accuracy and substantially improved F1-scores, indicating a more reliable representation of historic flooding patterns.



House Price Prediction Models
=============================

All England median
------------------

This model assumes all house price data can be imputed with the median house price for England. This is a baseline model to compare against. While it achieve a moderate error metric scores, it is not a useful model for risk prediction, and has very little skill.


HousePricesXGBRegressor
-----------------------

A combination of unit and sector data were selected as the features. A robust scaler was applied to deal with the heavily skewed numerical features, and OneHotEncoding was applied only on the soilType with the rest of categorical features dropped due to the high number of unique values. 
A modified XGBoost regression model was utilized for this prediction due to (1) its ability to use a validation set during training, (2) good performance in terms of prediction and training time, and (3) the well-balanced training vs validation fit. 
Overall, the most significant improvement was due to applying a log-transformer on the target variable, which resulted in a reduction in the RMSE, and MAE errors. This is expected because the median house price had significant outliers (up to ~8 million GBP).


Local Authority Prediction Models
=================================

Modal Local Authority Model
---------------------------

This model assumes all unlabeled data belongs to the most common Local Authority in the available training data. This is a baseline model to compare against. In this case, it does not achieve good accuracy and is generally a poor model to use.

Arbitrary Location Analysis
---------------------------
We utilized KNN models to predict flood risk and Local Authority for arbitrary coordinates. Through Grid Search Cross-Validation, we optimized classification (k=5) and regression (k=15, distance-weighted) parameters. 
Relying solely on OSGB36 coordinates, our spatial models significantly outperformed the baseline by leveraging spatial autocorrelation. Additionally, we implemented a meshgrid algorithm to visualize continuous flood risk across the UK.
