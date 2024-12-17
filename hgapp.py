import pandas as pd
import numpy as np
import gradio as gr
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]
data = pd.read_csv(url, names=column_names, na_values=" ?", skipinitialspace=True)

# Preprocess the dataset
data.dropna(inplace=True)

# Separate features and target
X = data.drop(columns=["income"])
y = data["income"]

# Encode target variable
label_encoder_income = LabelEncoder()
y_encoded = label_encoder_income.fit_transform(y)  # ' <=50K' -> 0, ' >50K' -> 1

# Encode categorical features
label_encoders = {}
for column in X.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Initialize LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X.columns.tolist(),
    class_names=["<=50K", ">50K"],  # Map to the target encoding
    mode="classification"
)

# Prediction and Explanation Function
def predict_and_explain(*args):
    # Step 1: Create input data as a DataFrame
    input_data = pd.DataFrame([args], columns=X.columns)

    # Step 2: Encode categorical values (handle unseen labels)
    for column, le in label_encoders.items():
        if column in input_data:
            input_data[column] = input_data[column].apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            input_data[column] = le.transform(input_data[column])

    # Step 3: Scale the input data
    input_scaled = scaler.transform(input_data)

    # Step 4: Predict the class and get probabilities
    prediction = model.predict(input_scaled)[0]
    prediction_label = "<=50K" if prediction == 0 else ">50K"

    # Step 5: Explain prediction using LIME
    exp = explainer.explain_instance(input_scaled[0], model.predict_proba, labels=[prediction])

    # Generate explanation plot for the predicted class
    fig = exp.as_pyplot_figure(label=prediction)
    plt.tight_layout()
    plt.savefig("lime_explanation.png")
    plt.close()

    # Step 6: Return prediction result and LIME plot
    return prediction_label, "lime_explanation.png"

# Gradio Interface
inputs = [
    gr.Slider(minimum=18, maximum=90, value=25, label="Age"),
    gr.Dropdown(choices=["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                         "Local-gov", "State-gov", "Without-pay", "Never-worked"],
                value="Private", label="Workclass"),
    gr.Number(value=77516, label="Fnlwgt"),
    gr.Dropdown(choices=["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
                         "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
                         "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
                value="Bachelors", label="Education"),
    gr.Number(value=13, label="Education-num"),
    gr.Dropdown(choices=["Married-civ-spouse", "Divorced", "Never-married", "Separated",
                         "Widowed", "Married-spouse-absent", "Married-AF-spouse"],
                value="Never-married", label="Marital-status"),
    gr.Dropdown(choices=["Tech-support", "Craft-repair", "Other-service", "Sales",
                         "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                         "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                         "Transport-moving", "Priv-house-serv", "Protective-serv",
                         "Armed-Forces"],
                value="Other-service", label="Occupation"),
    gr.Dropdown(choices=["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative",
                         "Unmarried"],
                value="Not-in-family", label="Relationship"),
    gr.Dropdown(choices=["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other",
                         "Black"],
                value="White", label="Race"),
    gr.Dropdown(choices=["Female", "Male"], value="Male", label="Sex"),
    gr.Number(value=0, label="Capital-gain"),
    gr.Number(value=0, label="Capital-loss"),
    gr.Number(value=40, label="Hours-per-week"),
    gr.Dropdown(choices=["United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
                         "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece",
                         "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy",
                         "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland",
                         "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
                         "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
                         "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
                value="United-States", label="Native-country"),
]

outputs = [
    gr.Textbox(label="Prediction"),
    gr.Image(label="LIME Explanation")
]

app = gr.Interface(fn=predict_and_explain, inputs=inputs, outputs=outputs, title="Income Prediction with LIME Explanation")
app.launch()