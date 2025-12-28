# Credit Score Classification System

This project focuses on building a machine learning--based system to
classify individuals into different credit score categories using their
financial and banking-related information. A Flask web application is
developed to provide a simple and interactive user interface for
prediction.

------------------------------------------------------------------------

## 📌 Problem Statement

Financial institutions collect large amounts of customer credit-related
data. Manually evaluating credit scores is time-consuming and
inefficient.

The objective of this project is to: - Automate credit score
classification - Reduce manual effort - Improve decision-making using
machine learning models

------------------------------------------------------------------------

## 🎯 Objectives

-   Train multiple machine learning models for credit score
    classification
-   Compare model performance
-   Deploy the best-performing models using Flask
-   Provide a clean and user-friendly web interface

------------------------------------------------------------------------

## 🗂️ Project Structure

    Credit-Score-Classification/
    │
    ├── app.py
    ├── credit-score-classification-multi-model.ipynb
    ├── templates/
    │   └── index.html
    ├── static/
    │   └── style.css
    ├── .gitignore
    └── README.md

> **Note:** Trained model files (`.pkl`) and dataset files are excluded
> due to GitHub file size limitations.

------------------------------------------------------------------------

## 🧠 Machine Learning Models Used

-   Logistic Regression
-   Random Forest Classifier

------------------------------------------------------------------------

## 📊 Features Used for Prediction

-   Age
-   Annual Income
-   Monthly Inhand Salary
-   Number of Bank Accounts

------------------------------------------------------------------------

## 🖥️ Web Application

-   Built using Flask
-   Allows users to input financial details
-   Supports multiple model selection
-   Displays predicted credit score category

------------------------------------------------------------------------

## 🚀 How to Run the Project

### Clone the Repository

    git clone https://github.com/mahins32/Credit-Score-Classification.git

### Install Dependencies

    pip install flask numpy pandas scikit-learn

### Run the Application

    python app.py

Open in browser:

    http://127.0.0.1:5000/

------------------------------------------------------------------------

## 🏁 Conclusion

This project demonstrates how machine learning can be used to automate
credit score classification using a Flask-based web interface.

------------------------------------------------------------------------

## 👨‍💻 Author

Mahin
