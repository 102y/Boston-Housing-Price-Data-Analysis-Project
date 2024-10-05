# Boston-Housing-Price-Data-Analysis-Project
Introduction
In this project, we analyze the Boston Housing Price dataset using several machine learning techniques such as Linear Regression, Support Vector Machines (SVM), Random Forest, and Artificial Neural Networks (ANN) using the PyTorch library. The goal is to build robust models to predict house prices based on a set of features.

Tools Used:
Numpy: For numerical data processing.
Pandas: For data analysis and cleaning.
Matplotlib: For data visualization.
Scikit-learn:
Splitting the data into training and testing sets.
Scaling the data using StandardScaler.
Building Linear Regression, SVM, and Random Forest models.
PyTorch: For creating and training Artificial Neural Network models.
Project Steps:
Loading the Data: We load the housing price data from a CSV file.
Data Visualization: Using Matplotlib to visualize the distribution of different features with respect to house prices.
Data Splitting: Splitting the data into features and target (prices), and then further dividing it into training and testing sets.
Data Scaling: Using StandardScaler to scale the data before feeding it into the models.
Model Building:
Linear Regression.
SVM.
Random Forest.
Artificial Neural Networks (ANN) with different layers.
Training the Models: Training all models on the scaled data.
Model Evaluation: Calculating the accuracy of each model on the test dataset.
Artificial Neural Networks:
Two different ANN models were built:

ANN (Model 1): Consisting of two hidden layers (64 and 32 neurons).
ANN (Model 2): Consisting of larger hidden layers (128 and 64 neurons).
Model Results:
Linear Regression: Accuracy is calculated using the .score() method.
SVM: Accuracy is calculated using SVR.
Random Forest: Accuracy is calculated using RandomForestRegressor.
Artificial Neural Networks: Accuracy is calculated using Mean Squared Error (MSE) and comparing it to the variance of the data.
How to Use:
Install Requirements: Make sure to install the necessary libraries like numpy, pandas, matplotlib, scikit-learn, torch.
Run the Project: Once the requirements are installed, you can run the project in any Python-supported environment.
Program Output:
The program will display the accuracy of each trained model:

Linear Regression Score
SVM Score
Random Forest Score
ANN Score (Model 1)
ANN Score (Model 2)
Project Goal:
The goal of this project is to present multiple models to predict house prices and analyze the performance of each model based on its accuracy in prediction.
