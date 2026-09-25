# 🎓 Graduate Admission Chance Predictor

### 🤖 Predicting Graduate Admission Chances with Machine Learning

A machine learning regression project that predicts a student's **chance of admission** to a graduate program using academic performance, university rating, statement/recommendation scores, CGPA, and research information.

The project uses **Linear Regression** and includes data cleaning, missing-value handling, duplicate removal, outlier treatment, exploratory analysis, model evaluation, and model serialization using Pickle.

---

## 🌟 About the Project

Graduate admission decisions can depend on several academic and profile-related factors.

This project explores the relationship between:

- 📚 GRE Score
- 📝 TOEFL Score
- 🏫 University Rating
- 📄 Statement of Purpose (SOP)
- 📑 Letter of Recommendation (LOR)
- 🎓 CGPA
- 🔬 Research experience

and the target variable:

- 🎯 **Chance of Admit**

The model predicts the admission chance as a continuous value rather than a simple yes/no classification.

---

## ✨ Key Features

### 📊 Data Analysis

The project performs exploratory analysis of the admission dataset, including:

- 🔎 Dataset inspection
- 📋 Data type analysis
- 🔁 Duplicate detection
- ❓ Missing-value analysis
- 📈 Statistical analysis
- 📊 Distribution visualization
- 📦 Boxplot-based outlier analysis
- 🔗 Correlation analysis

---

### 🧹 Data Preprocessing

The dataset is cleaned before training the model.

The preprocessing workflow includes:

- Removing duplicate records
- Filling missing SOP values using the mean
- Detecting outliers using the IQR method
- Handling identified outliers in LOR, CGPA, and Chance of Admit
- Removing the `Serial No.` column before model training

After duplicate removal, the dataset contains **400 records** and **8 columns**, including the target.

---

### 🤖 Machine Learning Model

The project uses:

**Linear Regression**

The model learns the relationship between the seven input features and the continuous `Chance of Admit` target.

The data is divided into:

- 🟢 **80% training data**
- 🔵 **20% testing data**

The split uses `random_state=1234`.

This results in:

- Training samples: **320**
- Testing samples: **80**

---

## 📋 Input Features

| Feature | Description |
|---|---|
| 📚 `GRE Score` | Graduate Record Examination score |
| 📝 `TOEFL Score` | Test of English as a Foreign Language score |
| 🏫 `University Rating` | Rating associated with the university |
| 📄 `SOP` | Statement of Purpose rating |
| 📑 `LOR` | Letter of Recommendation rating |
| 🎓 `CGPA` | Undergraduate cumulative GPA |
| 🔬 `Research` | Research experience indicator |

### 🎯 Target

```text
Chance of Admit
```

The target is a continuous value representing the estimated admission chance.

---

## 🔄 Machine Learning Workflow

```text
📂 Admission Dataset
        ↓
🔍 Data Inspection
        ↓
🔁 Duplicate Removal
        ↓
❓ Missing Value Handling
        ↓
📦 Outlier Detection & Treatment
        ↓
🔗 Correlation Analysis
        ↓
✂️ Train/Test Split
        ↓
🤖 Linear Regression
        ↓
🎯 Prediction
        ↓
📊 Model Evaluation
        ↓
💾 Save Model
```

---

## 📈 Model Evaluation

The Linear Regression model was evaluated using:

### 📉 Mean Squared Error (MSE)

**0.00452**

MSE measures the average squared difference between the predicted and actual admission chances.

### 📐 Root Mean Squared Error (RMSE)

**0.06723**

RMSE represents the prediction error in the same scale as the target variable.

### 🎯 R² Score

**0.71642**

The R² score indicates how much of the variation in the target variable is explained by the model on the test set.

---

## 🔗 Correlation Analysis

The project calculates a correlation matrix to examine relationships between the input variables and admission chance.

In the analyzed dataset, the strongest correlations with `Chance of Admit` include:

| Feature | Correlation with Chance of Admit |
|---|---:|
| 🎓 CGPA | 0.8598 |
| 📚 GRE Score | 0.7946 |
| 📝 TOEFL Score | 0.7829 |
| 🏫 University Rating | 0.7115 |
| 📄 SOP | 0.6864 |
| 📑 LOR | 0.6591 |
| 🔬 Research | 0.5485 |

These are correlations observed in the dataset and should not be interpreted as proof of causation.

---

## 📦 Dataset Information

The original dataset contains **402 rows**.

During preprocessing:

- 🔁 Duplicate records were identified and removed.
- ❓ Two missing values were found in the SOP feature and replaced with the SOP mean.
- 🧹 The resulting cleaned dataset contains **400 records**.

The project uses:

```text
admission.csv
```

---

## 🚨 Outlier Handling

The project uses the **Interquartile Range (IQR)** method to identify outliers.

Outliers were detected in:

- 📑 LOR
- 🎓 CGPA
- 🎯 Chance of Admit

The identified values were replaced with the corresponding feature mean according to the preprocessing logic used in the notebook.

---

## 💾 Model Saving

After training, the Linear Regression model is serialized using Python's Pickle module.

The saved model is:

```text
admission.pkl
```

The saved model can later be loaded and used to generate predictions without retraining the model.

---

## 🎯 Prediction

The trained model accepts the following seven input features:

- GRE Score
- TOEFL Score
- University Rating
- SOP
- LOR
- CGPA
- Research

The output is an estimated:

**🎓 Chance of Admission**

Example output from the notebook:

```text
Chance of admission: 0.54038151
```

The prediction value represents the model's estimated admission chance for the supplied input values.

---

## 🛠️ Technologies Used

- 🐍 **Python**
- 🐼 **Pandas**
- 🔢 **NumPy**
- 🤖 **Scikit-learn**
- 📊 **Matplotlib**
- 📈 **Seaborn**
- 💾 **Pickle**
- 📓 **Jupyter Notebook**

---

## 📁 Project Structure

```text
admission-predictor/
│
├── admission.csv
├── saif_admission.ipynb
├── admission.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

> The exact files in the repository may vary depending on whether the dataset and trained model are included.

---

## ▶️ How the Project Works

### 1️⃣ Load the Dataset

The admission dataset is loaded using Pandas.

### 2️⃣ Clean the Data

Duplicates, missing values, and identified outliers are handled.

### 3️⃣ Prepare Features

The `Serial No.` column is removed, and the remaining seven input features are separated from the target.

### 4️⃣ Split the Data

The cleaned dataset is divided into training and testing sets using an 80/20 split.

### 5️⃣ Train the Model

A Linear Regression model is fitted on the training data.

### 6️⃣ Generate Predictions

The trained model predicts admission chances for the test data.

### 7️⃣ Evaluate the Model

MSE, RMSE, and R² are calculated.

### 8️⃣ Save the Model

The trained model is stored as:

```text
admission.pkl
```

---

## 📊 Project Highlights

- 🎓 Graduate admission chance prediction
- 🤖 Linear Regression model
- 🧹 Data cleaning and preprocessing
- 🔁 Duplicate removal
- ❓ Missing-value handling
- 📦 IQR-based outlier detection
- 🔗 Correlation analysis
- 📈 Statistical and visual data analysis
- ✂️ 80/20 train-test split
- 📉 MSE: **0.00452**
- 📐 RMSE: **0.06723**
- 🎯 R²: **0.71642**
- 💾 Pickle-based model saving

---

## ⚠️ Disclaimer

This project is intended for **educational and machine learning demonstration purposes**.

The predicted admission chance is an estimate produced by a trained model and should not be treated as an official admission decision or guarantee.

---

## 👨‍💻 Project Information

| | |
|---|---|
| 📌 **Project Name** | Graduate Admission Chance Predictor |
| 🎯 **Task** | Admission Chance Prediction |
| 🤖 **Model** | Linear Regression |
| 📊 **Problem Type** | Regression |
| 📥 **Input Features** | 7 |
| 📤 **Target** | Chance of Admit |
| 🐍 **Language** | Python |
| 📓 **Development** | Jupyter Notebook |

---

## ⭐ Project Goal

The main goal of this project is to demonstrate how **machine learning regression** can be used to analyze academic and profile-related factors and estimate a student's graduate admission chance.
