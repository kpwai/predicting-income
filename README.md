# Predicting Income Using the Adult Dataset with Neural Networks and LIME Explainability

## Project Overview
This project uses the UCI Adult dataset to predict whether an individual's income exceeds $50K/year based on demographic data. The model utilizes a neural network for classification, and LIME (Local Interpretable Model-Agnostic Explanations) is used to explain the predictions made by the model.

## Objective
- Build a neural network model to classify whether an individual's income is greater than $50K/year.
- Use LIME to explain the predictions of the model.

## Dataset
The project uses the [UCI Adult dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data), which includes demographic and social factors such as age, work class, education, occupation, marital status, and race. The target variable (`income`) indicates whether the income is `<=50K` or `>50K`.

### Features:
- Age
- Work Class
- Education
- Marital Status
- Occupation
- Race
- Gender
- Capital Gain and Loss
- Hours per Week

## Techniques Used

### 1. **Data Preprocessing**
- Handled missing values by dropping rows with missing data.
- Encoded categorical variables using **LabelEncoder**.
- Standardized the numerical features using **StandardScaler** to improve model performance.

### 2. **Modeling**
- Implemented a **Multi-layer Perceptron (MLP)** neural network classifier with 100 and 50 units in the hidden layers.
- Trained the model on the preprocessed dataset and evaluated it using metrics like confusion matrix and classification report.

### 3. **Model Evaluation**
- Generated a confusion matrix to visualize the model’s predictions.
- Used **precision**, **recall**, **F1-score**, and **accuracy** to evaluate the model’s performance.

### 4. **Explainability with LIME**
- Applied **LIME** (Local Interpretable Model-Agnostic Explanations) to explain individual predictions.
- Visualized feature importance and generated local explanations for specific test instances.

## Installation
Clone this repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
```
