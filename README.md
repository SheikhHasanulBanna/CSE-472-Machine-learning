# CSE-472-Machine-learning


## Overview

This repository contains four assignments focusing on different areas of machine learning and linear algebra:

1. **Assignment 1**: Eigen Decomposition and Matrix Transformation
2. **Assignment 2**: Logistic Regression and AdaBoost for Classification
3. **Assignment 3**: Feed-forward Neural Networks
4. **Assignment 4**: PCA and EM Algorithm for GMM

## Assignment 1: Eigen Decomposition and Matrix Transformation

### Tasks
- **Eigen Decomposition**: Decompose a matrix into eigenvalues and eigenvectors.
- **Matrix Transformation**: Implement various matrix transformations such as rotation, scaling, and translation.

### Steps
1. Calculate eigenvalues and eigenvectors of a given matrix.
2. Implement and apply different matrix transformations.
3. Verify decomposition and visualize transformations.

## Assignment 2: Logistic Regression and AdaBoost for Classification

### Tasks
- **Logistic Regression**: Implement a logistic regression classifier using gradient descent.
- **AdaBoost**: Integrate logistic regression within the AdaBoost algorithm to create a strong classifier.

### Steps
1. Preprocess datasets (handle missing values, normalize features).
2. Implement logistic regression and use gradient descent for optimization.
3. Integrate logistic regression as a weak learner in AdaBoost.
4. Evaluate the models and report performance metrics.

## Assignment 3: Feed-forward Neural Networks

### Tasks
- **Feed-forward Neural Network (FNN)**: Build an FNN from scratch with components like Dense Layer, ReLU, Dropout, and Softmax layers.
- **Backpropagation**: Implement the backpropagation algorithm for training.
- **Model Training**: Train the model using mini-batch gradient descent.
- **Dataset**: Use the EMNIST letters dataset for training and evaluation.

### Steps
1. **Building Components**: Write separate classes for Dense Layer, ReLU, Dropout, and Softmax layers.
2. **Training**: Implement backpropagation and train the model using mini-batch gradient descent.
3. **Dataset Handling**: Load and preprocess the EMNIST letters dataset.
4. **Model Preservation**: Save the trained model using pickle and write a script to load and use the model for predictions.
5. **Evaluation**: Report training loss, validation loss, training accuracy, validation accuracy, and validation macro-F1 score. Prepare graphs for different learning rates and models.

## Assignment 4: PCA and EM Algorithm for GMM

### Tasks
- **Principal Component Analysis (PCA)**: Reduce the dimensionality of data using PCA and project data points along the two most prominent principal axes.
- **Expectation-Maximization (EM) Algorithm**: Estimate the parameters of Gaussian Mixture Models (GMM) using the EM algorithm.

### Steps
1. **PCA**:
   - Perform PCA using Singular Value Decomposition (SVD) on datasets with more than two dimensions.
   - Plot the data points along the two principal axes.
2. **GMM Estimation**:
   - Apply the EM algorithm for different values of K (number of Gaussian distributions).
   - Run the algorithm multiple times with different initializations.
   - Plot the best value of log-likelihood for convergence against K.
   - Choose the appropriate K and plot the estimated GMM.
3. **Bonus Task**:
   - Modify the EM algorithm to plot the estimated GMM after each iteration.
