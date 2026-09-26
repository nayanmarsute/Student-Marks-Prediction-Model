# 🎓 Student Marks Prediction using Machine Learning

An end-to-end Machine Learning project that predicts a student's **final exam marks** from academic and study-related factors using **Multiple Linear Regression**, served through a custom **Streamlit** interface (Markly).

> **Dataset note:** The included dataset is a privacy-safe synthetic student-performance dataset created specifically for this educational project. It contains realistic academic variables, controlled missing values, duplicate records, and a few invalid/outlier values so the preprocessing workflow can be demonstrated reproducibly.

---

## Assignment Requirements Coverage

| Requirement | Implementation |
|---|---|
| Data Collection | `src/data_collection.py` generates and stores the raw dataset in `data/` |
| Preprocessing | Duplicate removal, domain validation, missing-value imputation, outlier clipping, categorical encoding and scaling |
| Feature Selection | `SelectKBest(f_regression)` inside the training pipeline |
| Model Building | `LinearRegression` in `src/train_model.py` |
| Train/Test Split | 80/20 split with `random_state=42` |
| Evaluation | MAE, MSE, RMSE and R² saved in `reports/metrics.json` |
| Model Saving | Complete preprocessing + feature selection + model pipeline saved with Joblib |
| Streamlit UI | `app.py` |
| Prediction | Interactive final-marks prediction, input summary, and a "what-if" scenario simulator |
| Interpretation | Model-driven "next steps" derived from live sensitivity analysis, plus R² explanation and limitations below |
| Screenshots | `screenshots/` contains generated evaluation output; add real browser screenshots before submission |

---

## Features Used

- Study hours per day
- Attendance percentage
- Previous exam marks
- Assignment marks
- Internal marks
- Practice test score
- Sleep hours
- Study environment
- Internet quality

**Target:** Final exam marks (0–100)

---

## Project Structure

```text
Student-Marks-Prediction-Complete/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── student_marks_raw.csv
│   └── student_marks_clean.csv
├── models/
│   └── student_marks_pipeline.joblib
├── reports/
│   ├── metrics.json
│   ├── selected_features.csv
│   ├── test_predictions.csv
│   ├── actual_vs_predicted.png
│   └── metrics.png
├── screenshots/
│   ├── model_evaluation.png
│   └── README.md
└── src/
    ├── __init__.py
    ├── data_collection.py
    └── train_model.py
```

---

## 1. Data Collection

Run:

```bash
python src/data_collection.py
```

The script creates `data/student_marks_raw.csv` with 600 base records plus deliberate data-quality issues for demonstrating preprocessing.

## 2. Data Preprocessing

The training script:

- removes duplicate rows
- validates numeric domains
- converts invalid values to missing values
- handles missing values using pipeline imputers
- clips numeric outliers using the IQR rule
- one-hot encodes categorical variables
- standardizes numerical features
- performs feature selection using `SelectKBest`

Run:

```bash
python src/train_model.py
```

This also creates the cleaned dataset, evaluation reports, plots and serialized model pipeline.

## 3. Model Building

The required algorithm is **Multiple Linear Regression**. The preprocessing, feature selection and regression model are stored in one reproducible scikit-learn `Pipeline`, so the app and the training script always apply identical transformations.

## 4. Model Evaluation

The test set is kept separate from model fitting. The following metrics are calculated:

- **MAE:** average absolute prediction error
- **MSE:** average squared prediction error
- **RMSE:** square root of MSE, expressed in marks
- **R²:** proportion of variance in held-out marks explained by the model

The current training run produced:

| Metric | Test-set value |
|---|---:|
| MAE | 1.0515 |
| MSE | 1.6765 |
| RMSE | 1.2948 |
| R² | 0.8785 |

The exact values are also stored in `reports/metrics.json`. If the dataset or random seed changes, retrain and update the reported results from the generated file.

## 5. Model Saving

The complete pipeline is saved as:

```text
models/student_marks_pipeline.joblib
```

Saving the complete pipeline means the Streamlit app applies the same preprocessing steps used during training.

**Optional metadata:** `app.py` will automatically use two extra attributes on the saved pipeline object if present:

- `feature_ranges_` — a dict of `{feature_name: (min, max)}` from the training data, used to bound the input sliders to realistic values instead of arbitrary defaults.
- `residual_std_` — the standard deviation of residuals on the test set, used to show a realistic "±" margin next to the point estimate instead of a falsely precise single number.

Neither is required — the app falls back to sensible defaults if they're absent — but attaching them to the pipeline before saving is recommended and closes a known gap between what the model actually knows and what the UI currently assumes.

## 6. Streamlit Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

The application (`app.py`) currently provides:

- a routine-input form (study hours, attendance, exam history, assignments, internal marks, practice scores, sleep, environment, connectivity)
- a gated prediction flow — results and next steps only appear after the user submits the form, and the app flags when inputs have changed since the last estimate
- a predicted final-marks score with a performance band and a confidence margin
- an input summary card
- **model-driven "next steps":** rather than fixed, hand-picked thresholds, the app perturbs each numeric input against the live model and measures the actual effect on the prediction, then ranks and surfaces the levers the model responds to most — so the advice reflects what the trained model has learned, not assumptions baked into the UI
- a "what-if" simulator for study hours and practice score
- graceful handling of a missing or corrupted model file, and of categorical inputs the encoder wasn't trained on

**Not yet implemented** (mentioned as goals in earlier drafts of this project and worth adding before final submission if your assignment requires them in-app): a dedicated Model Evaluation tab displaying MAE/MSE/RMSE/R², a data explorer, a model-coefficient interpretation view, and an actual-vs-predicted plot rendered inside the app itself. These currently exist only as generated files in `reports/` — see Section 4 and the Screenshots section below. If your rubric requires them inside the running app rather than as static reports, they should be added as additional Streamlit tabs/pages before submission.

## 7. Interpretation

A high R² does **not** mean the model is "X% accurate." R² and accuracy are different concepts. MAE/RMSE should be interpreted in marks, while R² describes explained variance on the test set.

The model's coefficients can describe linear associations in the fitted data, but they should not automatically be interpreted as causal effects. The in-app "next steps" feature reflects this: it reports what the model is *sensitive to* given a specific student profile, not a claim about what *causes* better marks in general.

## Limitations

- The dataset is synthetic; real-world student data will have different distributions, correlations, and noise.
- Multiple Linear Regression assumes linear relationships between features and marks — nonlinear effects (e.g. diminishing returns on study hours) are only approximated.
- Slider bounds and the confidence margin are only as realistic as the training data's actual range — see the optional metadata note in Section 5.
- The sensitivity-based "next steps" reflect local behavior of the model around a student's specific inputs, not a global statement about which factor matters most for all students.

## Screenshots

Before submitting, run the Streamlit app and capture at least:

1. Prediction screen with a filled student profile and result.
2. The "what-if" simulator with a changed scenario.
3. Model Evaluation output (from `reports/metrics.json` or a dedicated tab, if added).
4. Actual-vs-predicted plot (`reports/actual_vs_predicted.png`).

Place those images in `screenshots/` and commit them to GitHub.


## Academic Honesty / Attribution

If this project is adapted from an existing repository or dataset, retain the relevant license and attribution. Do not claim another author's code, dataset or trained model as original work.

## License

For the synthetic dataset and original project code, use an appropriate license for your submission. If you incorporate third-party code or data, preserve its licensing terms.

---

## Prediction Verification

The Streamlit prediction page uses the saved `student_marks_pipeline.joblib` artifact directly.

The application creates a one-row DataFrame from the form inputs, in the same feature order used during training, and executes:

```python
model.predict(student_input)
```

The pipeline contains preprocessing, feature selection and the required Linear Regression model, so the UI is not using a manually coded marks formula.

If prediction fails on an unseen category (e.g. an encoder without `handle_unknown="ignore"`), the app retries once against safe fallback category values rather than crashing, and surfaces a clear error if the model file itself is missing or fails to load.

Before release, the prediction path should be checked with contrasting student profiles so that different inputs produce meaningfully different model outputs.

## UI

The Streamlit interface is a custom academic-analytics dashboard rather than a default Streamlit form. As currently implemented, it includes:

- a routine-input dashboard with a gated, session-aware prediction flow
- a results view with performance signal and confidence margin
- model-driven next steps based on live sensitivity analysis rather than hardcoded rules
- a what-if simulator

As noted in Section 6, a data explorer, an in-app model-evaluation tab, and a coefficient-interpretation view are natural extensions of this dashboard but are not yet part of `app.py` — they currently live as generated files under `reports/`.

The application intentionally avoids treating R² as "accuracy"; R² measures the proportion of variance explained by the fitted model on the evaluation data, and is presented that way throughout this README and the app.