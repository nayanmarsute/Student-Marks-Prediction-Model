# 🎓 Student Marks Prediction using Machine Learning

An end-to-end Machine Learning project that predicts a student's **final exam marks** from academic and study-related factors using **Multiple Linear Regression**, with an interactive **Streamlit** interface.

> **Dataset:** A privacy-safe synthetic student-performance dataset created for this educational project. It includes controlled missing values, duplicates, and invalid/outlier values to demonstrate preprocessing.

---

## 📌 Project Features

- 📊 Data collection and preprocessing
- 🧹 Missing-value handling
- 🔍 Duplicate and outlier detection
- 🔤 Categorical variable encoding
- 🎯 Feature selection using `SelectKBest`
- 🤖 Multiple Linear Regression
- 📊 80/20 train-test split
- 📈 MAE, MSE, RMSE and R² evaluation
- 💾 Saved ML pipeline using Joblib
- 🖥️ Interactive Streamlit prediction interface
- 📝 Student input summary and prediction
- 🔄 What-if prediction simulation
- 💡 Model-driven next-step suggestions

---

## 📊 Features Used

| Feature | Description |
|---|---|
| **Study Hours** | Average study hours per day |
| **Attendance** | Attendance percentage |
| **Previous Exam Marks** | Previous examination performance |
| **Assignment Marks** | Assignment performance |
| **Internal Marks** | Internal assessment marks |
| **Practice Test Score** | Practice/mock test performance |
| **Sleep Hours** | Average daily sleep |
| **Study Environment** | Student's study environment |
| **Internet Quality** | Internet access quality |

**Target:** Final exam marks (0–100)

---

## ⚙️ How to Run
**1. Install Dependencies**

Install all required Python packages:

pip install -r requirements.txt

If the above command doesn't work:

python -m pip install -r requirements.txt

**2. Generate the Dataset**

Run the data collection script:

python src/data_collection.py

This creates the raw student-performance dataset inside the data/ directory.

**3. Train the Model**

Run:

python src/train_model.py

This performs:

Data preprocessing
Missing-value handling
Duplicate and outlier handling
Feature selection
Model training
Model evaluation
Report generation
Model saving

The trained model is saved to:

models/student_marks_pipeline.joblib

**4. Run the Streamlit Application**

Start the application:

streamlit run app.py

If this doesn't work:

python -m streamlit run app.py

The application will open in your browser.

## 🖥️ Streamlit Application

Users can enter:

- Study hours
- Attendance
- Previous exam marks
- Assignment marks
- Internal marks
- Practice test score
- Sleep hours
- Study environment
- Internet quality

The application generates a predicted final score and provides a simple interpretation of the student's profile.

It also includes a What-If simulator for exploring changes in study-related inputs.

## 💾 Model Saving

The complete preprocessing and prediction pipeline is saved using Joblib:

models/student_marks_pipeline.joblib

This ensures that the same preprocessing used during training is applied when making predictions.

## 📷 Screenshots

![Uploading Screenshot 2024-12-22 213542.png…]()


## 🔄 Machine Learning Workflow

```text
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
