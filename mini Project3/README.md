# ML 2024 Mini Project 3

Welcome to the Mini Project 3 repository. This project focuses on implementing and evaluating various machine learning models on different datasets, emphasizing data preprocessing, model training, evaluation, and visualization.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Q1: Iris Dataset Analysis and SVM Models](#q1-iris-dataset-analysis-and-svm-models)
4. [Q2: Credit Card Fraud Detection](#q2-credit-card-fraud-detection)


## Overview

This project is divided into two main questions (Q1 and Q2), each focusing on different machine learning tasks:
1. Analyzing the Iris dataset and implementing Support Vector Machine (SVM) models.
2. Detecting credit card fraud using neural networks and SMOTE for data balancing.[Paper](https://arxiv.org/abs/1908.11553)

For more detailed information, please refer to the accompanying report file.

## Requirements

Ensure you have the following installed:
- Python 3.x
- Jupyter Notebook
- Required Python packages: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`, `imblearn`, `cvxopt`

Install the required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow imblearn cvxopt
```

### Q1: Iris Dataset Analysis and SVM Models

#### Section 1: Data Exploration and Visualization
- Dataset: Iris dataset.

##### Tasks:
- Loading the dataset and creating a DataFrame.
- Conducting statistical analysis and visualization using histograms, boxplots, and correlation matrices.
- Applying t-SNE and PCA for dimensionality reduction and visualization.

#### Section 2: SVM with Linear Kernel

##### Tasks:
- Splitting the dataset into training and testing sets.
- Standardizing the data.
- Training an SVM with a linear kernel.
- Evaluating the model using confusion matrices and classification reports.
- Visualizing decision boundaries using PCA-transformed data.

#### Section 3: SVM with Polynomial Kernel

#### Tasks:
- Training and evaluating SVM models with polynomial kernels of varying degrees.
- Comparing the performance of different polynomial degrees.
- Visualizing decision boundaries for each degree using PCA-transformed data.

#### Section 4: Custom SVM Implementation with Various Kernels

##### Tasks:

- Implementing SVM models from scratch with different kernel functions: linear, polynomial, RBF, and sigmoid.
- Training and evaluating the custom SVM models.
- Visualizing decision boundaries and comparing with the sklearn implementation.


### Q2: Credit Card Fraud Detection

#### Section 1: Data Preprocessing

Dataset: Credit card transactions dataset.

##### Tasks:

- Loading the dataset and performing initial preprocessing.
- Standardizing the 'Amount' feature.
- Splitting the data into training, validation, and testing sets.
- Visualizing the class distribution.

#### Section 2: SMOTE for Data Balancing

##### Tasks:
- Applying SMOTE to balance the training data.
- Visualizing the effect of SMOTE on the class distribution.

#### Section 3: Autoencoder for Noise Reduction

##### Tasks:

- Adding Gaussian noise to the data.
- Defining and training an autoencoder for noise reduction.
- Denoising the training, validation, and test data.

#### Section 4: Neural Network Classifier

##### Tasks:

- Defining and training a neural network classifier.
- Evaluating the classifier on the test set.
- Adjusting the classification threshold and analyzing its impact on recall and accuracy.
- Visualizing the confusion matrix and classification report.

#### Section 5: Experimenting with Different SMOTE Strategies

##### Tasks:

- Experimenting with different SMOTE sampling strategies.
- Evaluating the performance of the classifier with each strategy.
- Visualizing the recall and accuracy for different SMOTE strategies.


For more detailed information on the implementation, results, and analysis, please refer to the accompanying report file.
