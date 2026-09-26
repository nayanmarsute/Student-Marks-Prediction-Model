# Student Marks Prediction System

An end-to-end Machine Learning project that predicts a student's expected academic marks based on academic performance, study habits, attendance, and learning environment.

The project includes the complete Machine Learning workflow — data preprocessing, feature selection, model training, evaluation, model saving, and deployment through an interactive Streamlit application.

---

## 📌 Project Overview

Student academic performance can be influenced by several factors such as study time, attendance, previous examination performance, assignments, internal assessments, practice tests, sleep, and the learning environment.

This project uses these factors to build a Machine Learning model capable of estimating a student's expected marks.

The system provides a simple web interface where students can enter their academic and routine information and receive an estimated marks prediction along with an easy-to-understand interpretation.

---

## 🎯 Objectives

- Analyze factors affecting student academic performance.
- Clean and preprocess student performance data.
- Handle missing values, duplicate records, and outliers.
- Convert categorical variables into machine-readable features.
- Select relevant features for prediction.
- Build a Linear Regression model.
- Evaluate the model using standard regression metrics.
- Save the trained Machine Learning pipeline.
- Deploy the model using Streamlit.
- Provide an interactive interface for making predictions.

---

## ✨ Features

- 📊 Student performance data analysis
- 🧹 Missing-value handling
- 🔍 Duplicate record detection
- ⚠️ Outlier detection and handling
- 🔤 Categorical feature encoding
- 🎯 Feature selection
- 🤖 Linear Regression model
- 📈 Model performance evaluation
- 💾 Saved Machine Learning pipeline
- 🖥️ Interactive Streamlit interface
- 🔮 Real-time marks prediction
- 💡 Simple interpretation of prediction results

---

## 📂 Input Features

The model uses the following student-related features:

| Feature | Description |
|---|---|
| `study_hours` | Average hours spent studying per day |
| `attendance_percentage` | Student attendance percentage |
| `previous_exam_marks` | Marks obtained in a previous examination |
| `assignment_marks` | Assignment performance |
| `internal_marks` | Internal assessment marks |
| `practice_test_score` | Practice/mock test performance |
| `sleep_hours` | Average daily sleep duration |
| `study_environment` | Student's study environment |
| `internet_quality` | Quality of internet access |

### Target

The target variable is the student's predicted academic marks.

---
## 💡Screenshots

<img width="1846" height="882" alt="Screenshot 2026-09-26 181423" src="https://github.com/user-attachments/assets/f34cbbc9-3c16-4fe2-83de-f20af809464e" />


## 🔄 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Collection
     ↓
Data Cleaning
     ↓
Missing Value Handling
     ↓
Duplicate Detection
     ↓
Outlier Detection
     ↓
Categorical Encoding
     ↓
Feature Selection
     ↓
Train/Test Split
     ↓
Linear Regression
     ↓
Model Evaluation
     ↓
Save Trained Pipeline
     ↓
Streamlit Deployment
     ↓
Student Prediction
