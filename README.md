# Password Strength Detection using NLP and Machine Learning

## Author
Suyog Shrestha

Data Science & Business @ Knox College - June 2027

---

## Overview

This project is an **end-to-end machine learning system** that predicts the strength of a password using Natural Language Processing (NLP) and statistical feature engineering. The model classifies passwords into three categories:
- Weak
- Medium
- Strong

In addition to prediction, the system provides:
- Model confidence scores
- Class-wise probabilities
- A **Streamlit web application** for real-time user interaction

This project demonstrates the complete ML lifecycle: data preprocessing, feature engineering, model training, evaluation, hyperparameter tuning, deployment, and explainability.

---

## How to Run This Project 

To run the notebook:
1. Clone the repository
2. Install dependencies: 
    ```bash
    pip install -r requirements.txt
    ```
3. Launch Jupyter Notebook
4. Run all cells sequentially in the following sequence:
    - Notebooks/01_preprocessing.ipynb
    - Notebooks/02_modelling.ipynb

To run the app locally:
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Open terminal in the project directory
3. Run the command: 
    ```bash
    streamlit run app.py
    ```

---

## Streamlit Web Application

A real-time interactive web application was built using **Streamlit** to deploy the trained model.
The app applies the exact same feature pipeline used during training, ensuring consistency between offline evaluation and deployment.

---

## Modeling Approach

The problem is formulated as a **multi-class classification task** with three output classes: Weak, Medium, and Strong.
The following models were trained and compared:
1. Logistic Regression
2. Linear Support Vector Machine (SVM)
3. Multinomial Naive Bayes
4. XGBoost (tested but excluded due to suspiciously perfect performance)

The final model selected was Linear SVM, as it achieved the best balance of **high accuracy, stable generalization, and lower risk of overfitting.**

**Hyperparameter tuning** was performed using GridSearchCV, optimizing the regularization parameter C.

---

## Key Findings and Evaluation

Model performance was evaluated using:
    1. Accuracy
    2. Precision, Recall, F1-score
    3. Confusion Matrix

The following table shows the comparison of models trained and evaluated:

| Model | Accuracy | F1_Weighted | F1_Macro |
|------|---------|------|-----|
| **Linear SVM** | **0.98140** | **0.981302** | **0.970950** |
| Logistic Regression | 0.99330	| 0.993283 | 0.989683 |
| Naive Bayes | 0.83045 | 0.769224 | 0.582627 |

**Linear SVM** achieved the best performance and was selected as the final model. It demonstrated **Strong discrimination between classes, very low misclassification rate, and consistent performance across all strength categories.**

---

## Future Work

Potential extensions of this project include:
- Deep learning models (LSTM / Transformers)
- Integration with real-time security systems
- Adversarial testing on leaked password datasets
- REST API deployment using FastAPI