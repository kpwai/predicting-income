import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

data = pd.read_csv(url, names=column_names, na_values=' ?', skipinitialspace=True)

# Handle missing values
data = data.dropna()

# Encode categorical features
categorical_features = data.select_dtypes(include=['object']).columns
data[categorical_features] = data[categorical_features].apply(LabelEncoder().fit_transform)

# Split dataset into features and labels
X = data.drop('income', axis=1)
y = data['income']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the neural network classifier
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Feature importance
feature_importances = np.mean(np.abs(model.coefs_[0]), axis=1)
feature_names = X.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Neural Network Model')
plt.show()

# Predictions
y_pred = model.predict(X_test)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Initialize LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=X.columns.tolist(),
    class_names=['<=50K', '>50K'],
    discretize_continuous=True
)

# Explain a prediction
i = 0  # Index of the test instance to explain
exp = explainer.explain_instance(X_test[i], model.predict_proba, num_features=5)

# Save the explanation as an HTML file and open it in a browser
exp.save_to_file('lime_explanation_NN_adult.html')
webbrowser.open('lime_explanation_NN_adult.html')