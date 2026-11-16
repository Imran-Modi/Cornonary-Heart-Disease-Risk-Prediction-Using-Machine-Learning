# Heart Disease Risk Prediction Using Machine Learning

### Project Overview

This project presents a Machine Learning-based Heart Disease Risk Prediction System that predicts the risk of developing Coronary Heart Disease (CHD) within the next 10 years using patient health information.

The project uses the Framingham Heart Study dataset and implements five machine learning classification algorithms. A Streamlit web application provides an interactive interface for exploring data, comparing model performance, and generating predictions.

**Project Type:** Machine Learning | Classification | Healthcare Analytics

---

### Table of Contents

- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Tools & Technologies](#tools--technologies)
- [Data Cleaning & Preparation](#data-cleaning--preparation)
- [Machine Learning Models](#machine-learning-models)
- [Model Performance](#model-performance)
- [Streamlit Application](#streamlit-application)
- [Project Structure](#project-structure)
- [Installation & Execution](#installation--execution)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)
- [Conclusion](#conclusion)
- [Author](#author)

---

### Problem Statement

Heart disease is a major health concern worldwide. Early identification of individuals who may be at risk of developing coronary heart disease can support preventive healthcare and awareness.

The objective of this project is to develop a machine learning system that predicts the 10-year risk of coronary heart disease using patient demographic and health-related information.

The system compares multiple machine learning algorithms and provides an interactive Streamlit dashboard for model evaluation and prediction.

---

### Project Objectives

- Analyze and preprocess the Framingham Heart Study dataset.
- Handle missing values and class imbalance.
- Train five machine learning classification algorithms.
- Compare models using different evaluation metrics.
- Develop an interactive Streamlit web application.
- Generate predictions using patient health information.
- Visualize confusion matrices and ROC curves.

---

### Dataset

The project uses the **Framingham Heart Study Dataset**.

**Dataset Overview:**

| Attribute | Details |
|---|---|
| Dataset Name | Framingham Heart Study |
| Total Records | 4,240 |
| Input Features | 15 |
| Target Variable | Ten_Year_CHD_Risk |
| Training Records | 3,392 |
| Testing Records | 848 |
| Train-Test Split | 80:20 |
| File Format | CSV |

**Input Features:**

- Gender
- Age
- Education Level
- Current Smoker
- Cigarettes Per Day
- BP Medication
- Previous Stroke
- Hypertension
- Diabetes
- Total Cholesterol
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Body Mass Index (BMI)
- Heart Rate
- Blood Sugar Level

**Target Variable:**

`Ten_Year_CHD_Risk`

- 0 = No recorded CHD event
- 1 = CHD event recorded

**Target Class Distribution:**

| Target Class | Records |
|---|---:|
| No CHD Event (0) | 3,596 |
| CHD Event (1) | 644 |
| Total | 4,240 |

---

### Tools & Technologies

| Tool / Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning algorithms |
| Imbalanced-learn | SMOTENC for class imbalance |
| Streamlit | Interactive web application |
| Jupyter Notebook | Model development and experimentation |
| Git | Version control |
| GitHub | Project hosting |

---

### Data Cleaning & Preparation

The following preprocessing steps were performed:

- Imported the Framingham dataset using Pandas.
- Inspected the first and last five records.
- Checked dataset dimensions, column names, and data types.
- Renamed columns for better readability.
- Separated independent variables (X) and dependent variable (y).
- Split the dataset into training and testing sets using an 80:20 ratio.
- Identified missing values in the dataset.
- Applied median imputation to numerical features.
- Applied most-frequent-value imputation to categorical features.
- Handled missing cigarette consumption values based on smoking status.
- Scaled numerical features using StandardScaler.
- Handled class imbalance using SMOTENC.
- Created preprocessing and classification pipelines.

---

### Machine Learning Models

Five machine learning classification algorithms were implemented:

**1. Logistic Regression**

Used as a classification algorithm to estimate the probability of developing coronary heart disease.

**2. Decision Tree**

Uses decision rules to classify individuals based on their health-related features.

**3. Random Forest**

Uses multiple decision trees to generate classification predictions.

**4. Support Vector Machine (SVM)**

Classifies observations by identifying an optimal decision boundary.

**5. Gradient Boosting**

Uses sequentially trained decision trees to improve classification performance.

---

### Model Performance

The models were evaluated using the following performance metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

**Performance Comparison:**

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6710 | 0.2549 | 0.6047 | 0.3586 | 0.6977 |
| SVM | 0.6875 | 0.2444 | 0.5039 | 0.3291 | 0.6617 |
| Gradient Boosting | 0.7205 | 0.2300 | 0.3566 | 0.2796 | 0.6511 |
| Random Forest | 0.7913 | 0.2966 | 0.2713 | 0.2834 | 0.6411 |
| Decision Tree | 0.7040 | 0.1856 | 0.2791 | 0.2229 | 0.5297 |

**Key Observations:**

- Logistic Regression achieved the highest ROC AUC of 0.6977.
- Logistic Regression achieved the highest positive-class recall of 0.6047.
- Random Forest achieved the highest accuracy of 0.7913.
- SVM achieved a ROC AUC of 0.6617.
- Model performance was evaluated using multiple metrics because the dataset has an imbalanced target distribution.

---

### Streamlit Application

The project includes an interactive Streamlit web application with the following sections:

**1. Overview**

- Dataset summary.
- Total records and features.
- Available machine learning models.
- Target class distribution.

**2. Compare Models**

- Model performance comparison table.
- Selectable evaluation metrics.
- Model comparison charts.

**3. Model Evaluation**

- Confusion matrix.
- ROC curve comparison.
- Model selection.

**4. Predict**

- Patient personal information.
- Health measurements.
- Prediction model selection.
- CHD risk prediction.

**5. Explore Data**

- Dataset exploration.
- Data analysis and visualizations.

**6. Save Model**

- Model-saving functionality.

---

### Application Screenshots

**1. Dashboard Overview**

![Dashboard Overview](CHD%20SS-01.png)

---

**2. Model Performance Comparison**

![Model Comparison](CHD%20SS-02.png)

---

**3. Confusion Matrix and ROC Curve**

![Model Evaluation](CHD%20SS-03.png)

---

**4. Heart Disease Risk Prediction**

![Prediction](CHD%20SS-04.png)

---

### Project Structure

```text
heart-disease-risk-prediction/
│
├── README.md
├── app.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── framingham.csv
│
├── notebooks/
│   └── heart_disease_analysis.ipynb
│
├── models/
│   └── saved_models/
│
├── screenshots/
│   ├── overview.png
│   ├── model_comparison.png
│   ├── model_evaluation.png
│   └── prediction.png
│
└── src/
    ├── preprocessing.py
    └── train_model.py
```

---

### Installation & Execution

Follow these steps to run the project on your local computer.

**Step 1: Clone the Repository**

```bash
git clone https://github.com/YOUR-USERNAME/heart-disease-risk-prediction.git
```

**Step 2: Navigate to the Project Folder**

```bash
cd heart-disease-risk-prediction
```

**Step 3: Create a Virtual Environment (Optional)**

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

**Step 4: Install Required Libraries**

```bash
pip install -r requirements.txt
```

**Step 5: Run the Streamlit Application**

```bash
streamlit run app.py
```

The application will open in your web browser.

---

### Future Improvements

- Improve prediction performance through hyperparameter tuning.
- Explore additional machine learning algorithms.
- Improve minority-class precision and recall.
- Add feature importance and explainable AI visualizations.
- Improve the user interface.
- Deploy the application online.

---

### Disclaimer

This project is developed for educational and research purposes only.

The predictions are generated by machine learning models and must not be considered medical advice, a clinical diagnosis, or a substitute for professional healthcare evaluation.

---

### Conclusion

The Heart Disease Risk Prediction project demonstrates how machine learning can be applied to healthcare-related data to estimate the 10-year risk of coronary heart disease.

Using the Framingham Heart Study dataset, five classification algorithms were implemented and evaluated using multiple performance metrics.

The Streamlit application provides an interactive platform for exploring the dataset, comparing machine learning models, visualizing evaluation results, and generating predictions.

This project demonstrates practical skills in Python programming, data preprocessing, machine learning, model evaluation, data visualization, and Streamlit application development.

---

### 👤Author
***Imran Modi***

⭐ If you find this project useful, please consider giving this repository a star!

