# A03_Sampling_mnd15105

## Assignment 3 – Undersampling, Oversampling and SMOTE to a toy dataset

## Overview
This assignment explores how class imbalance affects machine learning performance and evaluates whether sampling techniques can improve model results. Many real-world datasets contain far fewer observations in one class (for example fraud vs. non-fraud), which can cause models to perform poorly when predicting the minority class. To address this challenge, three resampling methods were applied to a toy dataset:

Random Undersampling

Random Oversampling

SMOTE (Synthetic Minority Oversampling Technique)

The goal is to compare model performance before and after applying these techniques to determine whether resampling improves classification results.

## Objectives
The objective of this assignment is to:
-Understand the impact of class imbalance on classification models.
-Apply undersampling, oversampling, and SMOTE techniques.
-Train and evaluate models using each sampling method.
-Compare performance metrics to determine if sampling improves prediction accuracy—especially for the minority class.
  
## What tools are used
- California Housing Dataset
- GitHub
- GitHub Desktop
- Visual Studio Code
- Windows OS

## Tasks Completed
The following steps were completed during this assignment:
-Data Preparation
-Loaded and explored a toy dataset with class imbalance.
-Split the dataset into training and testing sets.
-Baseline Model
-Built an initial classification model using the original (imbalanced) data.
-Recorded baseline performance metrics (accuracy, precision, recall, F1 score).
-Applied Sampling Techniques
-Implemented Random Undersampling to reduce the majority class size.
-Implemented Random Oversampling to increase the minority class size.
-Implemented SMOTE to generate synthetic minority class examples.
-Model Training & Evaluation
-Trained the same model on each resampled dataset.
-Evaluated and compared performance across all methods.

## Outcome
The analysis demonstrated that sampling techniques can significantly affect model performance.

Key observations:
-The baseline model typically favored the majority class.
-Undersampling improved minority detection but sometimes reduced overall accuracy due to information loss.
-Oversampling improved recall but risked overfitting.
-SMOTE often produced the best balance between precision and recall.

This confirms that sampling can improve model fairness and detection of minority cases, but results vary depending on the dataset and model.

## Expected output
From this assignment, we expect learners to:
-Understand why accuracy alone is not enough for imbalanced datasets.
-Recognize the trade-offs between different resampling techniques.
-Learn when sampling helps and when it may not.
-Develop the ability to evaluate models using appropriate metrics such as:

Precision

Recall
-F1 Score
-Confusion Matrix

## What the Audience Should Do to Run This Assignment

To replicate this assignment, the audience should:
-Install Required Libraries
-Python
-scikit-learn
-imbalanced-learn
-pandas
-numpy
-matplotlib or seaborn (optional for visualization)
-Prepare the Dataset
-Load the toy dataset.
-Split into training and testing sets.
-Train a Baseline Model
-Train a classifier using the original imbalanced data.
-Record evaluation metrics.
-Apply Sampling Techniques
-Use RandomUnderSampler
-Use RandomOverSampler

Use SMOTE
-Retrain and Compare
-Train the model on each resampled dataset.
-Compare results using recall, precision, and F1 score.
-Analyze Results
-Determine which sampling method performs best.
-Discuss trade-offs and practical implications.

## Notes
This repository will be updated with additional assignments and projects as the course progresses.
