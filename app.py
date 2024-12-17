import gradio as gr
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Load and preprocess the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
data = pd.read_csv(url, names=column_names, na_values=' ?', skipinitialspace=True)
data = data.dropna()

# Encode categorical features
categorical_features = data.select_dtypes(include=['object']).columns
data[categorical_features] = data[categorical_features].apply(LabelEncoder().fit_transform)

# Split dataset
X = data.drop('income', axis=1)
y = data['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Set up LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=X.columns.tolist(),
    class_names=['<=50K', '>50K'],
    discretize_continuous=True
)

# Define the prediction function
def predict(inputs):
    inputs = np.array(inputs).reshape(1, -1)
    prediction = model.predict(inputs)[0]
    return "Prediction: >50K" if prediction == 1 else "Prediction: <=50K"

# Define LIME explanation function
def explain(inputs):
    inputs = np.array(inputs).reshape(1, -1)
    exp = explainer.explain_instance(inputs[0], model.predict_proba, num_features=5)
    exp.save_to_file('lime_explanation.html')
    return "LIME explanation generated. Open 'lime_explanation.html' in your browser."

# Gradio interface
inputs = [
    gr.Number(label=col) for col in X.columns
]
output1 = gr.Textbox(label="Prediction")
output2 = gr.Textbox(label="LIME Explanation")

interface = gr.Interface(
    fn=lambda *args: (predict(args), explain(args)),
    inputs=inputs,
    outputs=[output1, output2],
    title="Income Prediction with LIME Explanation",
    description="Enter values for each feature to get a prediction and a LIME explanation."
)

interface.launch()