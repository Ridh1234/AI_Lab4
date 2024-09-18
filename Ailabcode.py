import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data'
col_headers = ['Diagnosis', 'Triiodothyronine', 'Thyroxine', 'TSH']
df = pd.read_csv(data_url, names=col_headers, delim_whitespace=True)

diagnosis_mapping = {1: 'Hyperthyroid', 2: 'Hypothyroid', 3: 'Normal'}
df['Diagnosis'] = df['Diagnosis'].map(diagnosis_mapping)

df.fillna('Unknown', inplace=True)

features = df.drop(columns=['Diagnosis'])
labels = df['Diagnosis']

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

training_data = features_train.copy()
training_data['Diagnosis'] = labels_train

network = BayesianNetwork([('Triiodothyronine', 'Diagnosis'), ('Thyroxine', 'Diagnosis'), ('TSH', 'Diagnosis')])

network.fit(training_data, estimator=MaximumLikelihoodEstimator)

inference_engine = VariableElimination(network)

def make_predictions(inference_model, test_features):
    predictions = []
    for _, instance in test_features.iterrows():
        evidence = instance.to_dict()
        try:
            prediction = inference_model.map_query(variables=['Diagnosis'], evidence=evidence)
            predictions.append(prediction['Diagnosis'])
        except KeyError:
            predictions.append('Unknown')
    return predictions

predicted_labels = make_predictions(inference_engine, features_test)

model_accuracy = accuracy_score(labels_test, predicted_labels)

print(f'Prediction Accuracy: {model_accuracy * 100:.2f}%')
