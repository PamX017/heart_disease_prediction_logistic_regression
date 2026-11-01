# Heart Disease Risk Prediction using Logistic Regression

## 📌 Project Overview
This project focuses on building a binary classification Machine Learning model using **Logistic Regression** to predict the likelihood of a patient having heart disease based on various clinical parameters. 

Developed as part of an RCA (Root Cause Analysis / Machine Learning) project, this implementation emphasizes model interpretability, quantitative evaluation, and ease of deployment using Python and `scikit-learn`.

---

## 📊 Dataset Description
The dataset used is derived from the **UCI Cleveland Heart Disease dataset** (available on [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)).

### Features Breakdown:
1. `age`: Age in years
2. `sex`: Sex (1 = male; 0 = female)
3. `cp`: Chest pain type (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic)
4. `trestbps`: Resting blood pressure (in mm Hg on admission to the hospital)
5. `chol`: Serum cholesterol in mg/dl
6. `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. `restecg`: Resting electrocardiographic results (0, 1, 2)
8. `thalach`: Maximum heart rate achieved
9. `exang`: Exercise-induced angina (1 = yes; 0 = no)
10. `oldpeak`: ST depression induced by exercise relative to rest
11. `slope`: The slope of the peak exercise ST segment
12. `ca`: Number of major vessels (0-3) colored by fluoroscopy
13. `thal`: Thalassemia type (0 = normal; 1 = fixed defect; 2 = reversible defect)
14. `target`: Diagnosis of heart disease (**0** = No Disease, **1** = Disease Present)

---

## ⚙️ Key Technical Workflow
1. **Data Cleaning & Deduplication:** Removed duplicate rows to prevent data leakage between train and test splits.
2. **Feature Scaling:** Applied `StandardScaler` to standardize feature ranges (mean = 0, variance = 1) ensuring uniform contribution to gradient optimization.
3. **Model Training:** Trained a `LogisticRegression` baseline classifier.
4. **Interpretability Analysis:** Extracted learned coefficients and calculated Odds Ratios ($\mathbf{OR} = e^{\beta}$) to determine key risk factors.
5. **Single-Sample Pipeline:** Constructed an end-to-end inference function for real-time risk evaluation of custom patient input values.

---

## 🚀 Getting Started

### Prerequisites
Ensure Python 3.8+ is installed along with the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Project
1. Clone this repository or download the source code.
2. Place `heart.csv` in the root directory.
3. Execute the script:

```bash
python main.py
```

---

## 📈 Model Performance & Results

| Metric | Score |
| :--- | :--- |
| **Accuracy** | ~80.33% |
| **Precision** | ~80.00% |
| **Recall** | ~84.85% |
| **F1-Score** | ~82.35% |
| **ROC-AUC Score** | ~0.8712 |

### Primary Risk Drivers
- **Chest Pain Type (`cp`):** Positively correlated with heart disease.
- **Max Heart Rate Achieved (`thalach`):** Higher values correlate with presence of target class.
- **ST Depression (`oldpeak`):** Strong negative indicator when elevated.

---

## 📐 Mathematical Foundation
Logistic Regression maps real-valued linear inputs to probabilities between $0$ and $1$ using the Sigmoid activation function:

$$\sigma(z) = \frac{1}{1 + e^{-z}} \quad \text{where} \quad z = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n$$

The loss function optimized during training is Binary Cross-Entropy (Log Loss):

$$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

---

## 📜 License
This project is for educational purposes as part of an RCA coursework submission.