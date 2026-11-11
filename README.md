# 🛡️ Phishing Email Detector

An AI/ML-based phishing email detection system that classifies emails as **Safe** or **Phishing** using textual and URL-based features.

The system uses **TF-IDF text features**, engineered **URL features**, and **Logistic Regression** to identify potentially malicious emails.

## 📌 Project Overview

Phishing emails are designed to trick users into revealing sensitive information such as passwords, financial details, or personal data.

This project develops a machine learning-based system that analyzes the content and URLs present in an email and predicts whether the email is:

- ✅ Safe
- 🚨 Phishing

The trained model is integrated into a web application where users can enter email content and receive a prediction along with model confidence.

## 🎯 Objectives

- Detect phishing emails automatically using machine learning.
- Analyze email text using TF-IDF features.
- Extract useful URL-based characteristics.
- Combine textual and URL features for improved classification.
- Evaluate the model using standard classification metrics.
- Provide an easy-to-use web interface.
- Deploy the trained model as a web application.

## ✨ Features

- 📧 Email text classification
- 🔗 URL feature extraction
- 🤖 Machine learning-based prediction
- 📊 Model confidence score
- 🛡️ Safe/Phishing classification
- 📈 Model evaluation
- 🧪 Automated tests
- 🌐 Web-based interface
- 🚀 Deployment-ready trained model

  ## 🧠 Machine Learning Approach
 1. Data Preparation
 2. Text Feature Extraction
 3. URL Feature Extraction

## 📊 Dataset
The project uses multiple email datasets for training and evaluation.
Due to their large size, the original datasets are not stored directly in this GitHub repository.
Download Dataset
🔗[ [Download Phishing Email Detector Datasets](https://drive.google.com/drive/folders/1mmoeQS7sDqexaU1TiBbcDUEzHSR80T_2?usp=sharing)]

The dataset folder contains:
phishing_email.csv
CEAS_08.csv
Enron.csv
Ling.csv
Nazario.csv
Nigerian_Fraud.csv
SpamAssasin.csv
emails.csv
Dataset Preparation



## How the Application Works
User enters email content.
The application preprocesses the text.
TF-IDF features are generated.
URL-related features are extracted.
Features are passed to the trained Logistic Regression model.
The model predicts the email class.
The application displays:
Prediction
Confidence score
Model information
  
