Regression Models

I focused on developing and evaluating regression models for predicting the popularity of Instagram posts. My main contributions included preprocessing, feature engineering, and implementing classical regression techniques.

Models Implemented

Linear Regression
Random Forest Regressor
Ridge Regression
Key Features

Designed a robust preprocessing pipeline tailored for Turkish language content.
Removed Turkish stopwords using NLTK and customized filters.
Engineered features based on post metadata, including follower counts, post counts, and engagement metrics.
Applied log transformations to handle skewed numeric distributions.
Used cross-validation for evaluating models and ensuring stability.
Tuned hyperparameters for models such as Random Forest and Ridge Regression.
Performance Metrics

Achieved Mean Squared Error (MSE) of XX using Ridge Regression.
Implemented log-transformed evaluation metrics such as log-MSE for better interpretability.
Assessed model performance with R² scores, achieving consistent accuracy across posts with varying popularity.
Code Structure

Preprocessing Functions:
Handled missing data and outliers.
Implemented log transformations for numeric fields like like_count and follower_count.
Extracted textual features using TF-IDF for captions and bios.
Model Training and Evaluation:
Trained classical regression models with optimized parameters.
Performed train-validation splits and computed metrics like MSE, RMSE, and R².
Prediction Pipeline:
Designed a scalable pipeline for generating predictions on test data.
Saved results in the required JSON format for submission. Project Files

project-explore.ipynb: Main notebook containing all regression models.
Feature Extraction Scripts: Preprocessing and log-transform pipelines.
Evaluation Functions: Custom metrics like log-MSE and R² calculations.
Results

Consistently achieved robust predictions across posts with varying engagement levels.
Successfully handled the challenges of skewed numeric distributions using log transformations.
Built a scalable and reusable prediction pipeline for generating test results in JSON format.
