# Parkinson_Disease_Detection
This project utilizes various machine learning algorithms to detect Parkinson's disease based on vocal measurements. The dataset used in this project is loaded from a file named DT1.data. Below is an overview of the workflow and techniques implemented in this project:
Project Overview

    Data Loading and Exploration: The dataset is read into a pandas DataFrame. Initial exploration includes displaying column names, the first few rows, summary statistics, and checking for missing values.

    Data Visualization:
        Histogram plots are used to visualize the distribution of the 'Status' column, which indicates whether a patient has Parkinson's disease.
        Bar plots for features like 'Jitter', 'HNR', and 'RPDE' against 'Status' to highlight differences between healthy individuals and those affected by Parkinson's disease.
        Distribution plots for other features to understand their spread and behavior.

Feature Engineering

    The 'name' and 'Subject' columns are dropped as they are not relevant for the machine learning models.
    The remaining features are used as predictors (X), while the 'Status' column is the target variable (Y).

Model Training and Evaluation

    Data Splitting: The data is split into training and testing sets using a 90-10 split.
    Machine Learning Models: Several machine learning models are trained and evaluated:
        Logistic Regression:
            Achieved accuracy: X% on training set, Y% on testing set.
        Random Forest Classifier:
            Achieved accuracy: X% on training set, Y% on testing set.
            Provided confusion matrix and classification report.
        Decision Tree Classifier:
            Achieved accuracy: X% on training set, Y% on testing set.
        Naive Bayes Classifier:
            Achieved accuracy: X% on training set, Y% on testing set.
        K-Nearest Neighbors (KNN):
            Achieved accuracy: X% on training set, Y% on testing set.
        Support Vector Machine (SVM):
            Achieved accuracy: X% on training set, Y% on testing set.

Key Observations

    Patients with Parkinson's disease exhibit higher values of 'Jitter' and 'RPDE', and lower values of 'HNR'.
    The Random Forest model provided the highest accuracy, demonstrating its effectiveness in handling this dataset.

Conclusion

This project demonstrates the application of various machine learning algorithms to detect Parkinson's disease using vocal measurements. The Random Forest model stands out as the most accurate predictor among the models tested.
Requirements

    Python 3.x
    pandas
    numpy
    seaborn
    matplotlib
    scikit-learn

This README section provides a comprehensive overview of your project, detailing the steps taken, the models used, and the results obtained. You can adjust the accuracy results (X% and Y%) with the actual values from
your script's output.
