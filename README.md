🎓 Student Marks Prediction using Machine Learning

An end-to-end Machine Learning project that predicts a student's final exam marks from academic and study-related factors using Multiple Linear Regression, with an interactive Streamlit interface.

Dataset: A privacy-safe synthetic student-performance dataset created for this educational project. It includes controlled missing values, duplicates, and invalid/outlier values to demonstrate preprocessing.

📌 Project Features

Data collection and preprocessing

Missing-value handling

Duplicate and outlier detection

Categorical variable encoding

Feature selection using SelectKBest

Multiple Linear Regression

80/20 train-test split

MAE, MSE, RMSE and R² evaluation

Saved ML pipeline using Joblib

Interactive Streamlit prediction interface

Student input summary and prediction

What-if prediction simulation

Model-driven next-step suggestions

📊 Features Used

Feature

Description

Study Hours

Average study hours per day

Attendance

Attendance percentage

Previous Exam Marks

Previous examination performance

Assignment Marks

Assignment performance

Internal Marks

Internal assessment marks

Practice Test Score

Practice/mock test performance

Sleep Hours

Average daily sleep

Study Environment

Student's study environment

Internet Quality

Internet access quality

Target: Final exam marks (0–100)

🔄 Machine Learning Workflow

Raw Dataset
    ↓
Data Cleaning & Validation
    ↓
Missing Value Handling
    ↓
Duplicate & Outlier Handling
    ↓
Categorical Encoding & Scaling
    ↓
Feature Selection
    ↓
Train/Test Split
    ↓
Multiple Linear Regression
    ↓
Model Evaluation
    ↓
Save Model Pipeline
    ↓
Streamlit Deployment

📈 Model Performance

Current test-set results:

Metric

Result

MAE

0.7781

MSE

0.9986

RMSE

0.9993

R² Score

0.9513

Average prediction error: approximately 0.78 marks

Typical prediction error: approximately 1 mark

R² indicates approximately 95.13% explained variance

R² is not the same as prediction accuracy percentage. MAE and RMSE are interpreted in marks, while R² measures explained variance.

📁 Project Structure

Student-Marks-Prediction-Complete/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── student_marks_raw.csv
│   └── student_marks_clean.csv
│
├── models/
│   └── student_marks_pipeline.joblib
│
├── reports/
│   ├── metrics.json
│   ├── selected_features.csv
│   ├── test_predictions.csv
│   ├── actual_vs_predicted.png
│   └── metrics.png
│
├── screenshots/
│
└── src/
    ├── __init__.py
    ├── data_collection.py
    └── train_model.py

⚙️ How to Run

1. Install dependencies

pip install -r requirements.txt 
If this dosen't work try
python -m pip install -r requirements.txt

2. Generate the dataset

python src/data_collection.py

3. Train the model

python src/training_model.py

This performs preprocessing, feature selection, training, evaluation, and saves the trained pipeline and reports.

The model is saved to:

models/student_marks_pipeline.joblib

6. Run the Streamlit application

streamlit run app.py
If this dosen't work try
python -m streamlit run app.py

The application will open in your browser.

🖥️ Streamlit Application

Users can enter:

Study hours

Attendance

Previous exam marks

Assignment marks

Internal marks

Practice test score

Sleep hours

Study environment

Internet quality

The application generates a predicted final score and provides a simple interpretation of the student's profile.

It also includes a What-If simulator for exploring changes in study-related inputs.

💾 Model Saving

The complete preprocessing and prediction pipeline is saved using Joblib:

models/student_marks_pipeline.joblib

This ensures the same preprocessing used during training is applied during prediction.

📷 Screenshots

Add screenshots of the running application to the screenshots/ folder.

Recommended screenshots:

Prediction interface

Prediction result

What-if simulator

Model evaluation

🚀 Future Improvements

Use a larger real-world student dataset

Add more academic and behavioral features

Compare additional regression algorithms

Add cross-validation and hyperparameter tuning

Add performance history and trend visualization

Deploy the application online

🎓 Project Information

Project Type: Machine Learning / Data Science Internship Project
Domain: Education & Academic Performance Prediction
Technologies: Python, Pandas, NumPy, Scikit-learn, Joblib, Streamlit
