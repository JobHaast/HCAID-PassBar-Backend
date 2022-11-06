import uuid
import numpy as np
import shap
import base64

import pandas as pd

from flask import Flask, request, jsonify
from keras.models import load_model
from sklearn.model_selection import train_test_split
from io import BytesIO
from flask_cors import CORS

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

app = Flask(__name__)
# Solves Cross Origin Resource Sharing (CORS)
CORS(app)

# Used for validation
EXPECTED = {
    "decile1": {"min": 1, "max": 10},
    "decile3": {"min": 1, "max": 10},
    "lsat": {"min": 120, "max": 180},
    "ugpa": {"min": 0, "max": 4},
    "fulltime": {},
    "grad": {},
}


# This method is used for data transformation.
def transform_data(df):
    df_zfy = df[['zfygpa', 'decile1']]
    condlist = []

    for i in range(1, 11):
        condlist.append(df_zfy[df_zfy['decile1'] == i]['zfygpa'].min())

    df['decile1'] = df.apply(lambda row: 9 if np.isnan(row['decile1']) & (row['zfygpa'] > .81) else row['decile1'],
                             axis=1)
    df['decile1'] = df.apply(lambda row: 8 if np.isnan(row['decile1']) & (row['zfygpa'] > .51) else row['decile1'],
                             axis=1)
    df['decile1'] = df.apply(lambda row: 7 if np.isnan(row['decile1']) & (row['zfygpa'] > .2) else row['decile1'],
                             axis=1)
    df['decile1'] = df.apply(lambda row: 6 if np.isnan(row['decile1']) & (row['zfygpa'] > -.02) else row['decile1'],
                             axis=1)
    df['decile1'] = df.apply(lambda row: 5 if np.isnan(row['decile1']) & (row['zfygpa'] > -.3) else row['decile1'],
                             axis=1)
    df['decile1'] = df.apply(lambda row: 4 if np.isnan(row['decile1']) & (row['zfygpa'] > -.58) else row['decile1'],
                             axis=1)
    df['decile1'] = df.apply(lambda row: 3 if np.isnan(row['decile1']) & (row['zfygpa'] > -.89) else row['decile1'],
                             axis=1)
    df['decile1'] = df.apply(lambda row: 2 if np.isnan(row['decile1']) & (row['zfygpa'] > -1.3) else row['decile1'],
                             axis=1)
    df['decile1'] = df.apply(lambda row: 1 if np.isnan(row['decile1']) & (row['zfygpa'] > -3.35) else row['decile1'],
                             axis=1)

    # Same as above but for decile3 and zgpa
    df_z = df[['zgpa', 'decile3']]
    condlist = []

    for i in range(1, 11):
        condlist.append(df_z[df_z['decile3'] == i]['zgpa'].min())

    df['decile3'] = df.apply(lambda row: 10 if np.isnan(row['decile3']) & (row['zgpa'] > 1.24) else row['decile3'],
                             axis=1)
    df['decile3'] = df.apply(lambda row: 9 if np.isnan(row['decile3']) & (row['zgpa'] > .83) else row['decile3'],
                             axis=1)
    df['decile3'] = df.apply(lambda row: 8 if np.isnan(row['decile3']) & (row['zgpa'] > .5) else row['decile3'], axis=1)
    df['decile3'] = df.apply(lambda row: 7 if np.isnan(row['decile3']) & (row['zgpa'] > .21) else row['decile3'],
                             axis=1)
    df['decile3'] = df.apply(lambda row: 6 if np.isnan(row['decile3']) & (row['zgpa'] > -.05) else row['decile3'],
                             axis=1)
    df['decile3'] = df.apply(lambda row: 5 if np.isnan(row['decile3']) & (row['zgpa'] > -.33) else row['decile3'],
                             axis=1)
    df['decile3'] = df.apply(lambda row: 4 if np.isnan(row['decile3']) & (row['zgpa'] > -.6) else row['decile3'],
                             axis=1)
    df['decile3'] = df.apply(lambda row: 3 if np.isnan(row['decile3']) & (row['zgpa'] > -.91) else row['decile3'],
                             axis=1)
    df['decile3'] = df.apply(lambda row: 2 if np.isnan(row['decile3']) & (row['zgpa'] > -1.34) else row['decile3'],
                             axis=1)
    df['decile3'] = df.apply(lambda row: 1 if np.isnan(row['decile3']) & (row['zgpa'] > -6.44) else row['decile3'],
                             axis=1)

    # Fill with column mean
    df['decile3'].fillna(5, inplace=True)
    df['decile1'].fillna(5, inplace=True)

    df['grad'] = df['grad'].replace(['Y', 'X', 'O'], [1, 0, 0])
    df['grad'] = df['grad'].fillna(1)

    df['fulltime'].fillna(1.0, inplace=True)
    df['fulltime'] = df['fulltime'].replace(2, 0)

    df[['decile3', 'decile1', 'lsat', 'grad', 'fulltime']] = df[
        ['decile3', 'decile1', 'lsat', 'grad', 'fulltime']].astype(int)

    return df


# This method is used for loading the needed data.
def load_data():
    # Fetch data from csv file.
    data = pd.read_csv('./data/bar_pass_prediction.csv')

    # Transform data
    df = transform_data(data)

    # Only take the needed column(s)
    df = df[['decile3', 'decile1', 'lsat', 'ugpa', 'fulltime', 'grad', 'pass_bar']]

    # Shuffle data to prevent none existent order corr
    df = df.sample(frac=1, random_state=1)

    # Split dataset into target variable and features
    df_y = df['pass_bar']
    df_X = df.drop('pass_bar', axis=1)

    # Split the data into a train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.3, random_state=42)

    return (X_train, y_train), (X_test, y_test)


# The setup method is used for setting everything up that we need to work with
def setup():
    # Load neural network
    _model = load_model("./exported-h5/binaryClassificationBarPass.h5")

    # Load training data
    (train_features, _), _ = load_data()

    # Load SHAP (Explainability AI)
    _shap_explainer = shap.KernelExplainer(_model, train_features[:100])

    return _model, _shap_explainer


(model, shap_explainer) = setup()


@app.route('/', methods=['GET'])
def root():
    response = {
        "author": "Job Haast",
        "description": "API for PassBar",
        "made possible by": "Tim van den Enden",
        "version": "1.0"
    }
    return jsonify(response)


@app.route('/api/predict', methods=['POST'])
def predict():
    content = request.json
    errors = []

    # # Check for valid input fields
    # for name in content:
    #     if name in EXPECTED:
    #         expected_min = EXPECTED[name]['min']
    #         expected_max = EXPECTED[name]['max']
    #         value = float(content[name])
    #         if value < expected_min or value > expected_max:
    #             errors.append(
    #                 f"Out of bounds: {name}, has value of: {value}, but should be between {expected_min} and {expected_max}."
    #             )
    #     else:
    #         errors.append(f"Unexpected field: {name}.")

    # Check for missing input fields
    for name in EXPECTED:
        if name not in content:
            errors.append(f"Missing value: {name}.")

    if len(errors) < 1:
        # Predict
        x = np.zeros((1, 6))

        x[0, 0] = content['decile3']
        x[0, 1] = content['decile1']
        x[0, 2] = content[
                      'lsat'] - 120  # The range of values of the lsat in this dataset do not correspond with real-life data. In this dataset it ranges from 0-60 (educated guess). The range from the real lsat is 120-180 (which comes to 60 inbetween). So to calculate an accurate lsat score for a prediction 120 points must be deducted from provided real-life lsat score.
        x[0, 3] = content['ugpa']
        x[0, 4] = content['fulltime']
        x[0, 5] = content['grad']

        # Prediction
        prediction = model.predict(x)
        chance_of_passing = float(prediction[0]) * 100

        # Explanation
        shap_values = shap_explainer.shap_values(x)
        shap_plot = shap.force_plot(
            shap_explainer.expected_value,
            shap_values[0],
            x,
            matplotlib=True,
            feature_names=['Decile3', 'Decile1', 'lsat', 'ugpa', 'fulltime', 'grad'],
            show=False,
            plot_cmap=['#77dd77', '#f99191']
        )

        # Encode shap img into base64,
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches="tight", transparent=True)
        shap_img = base64.b64encode(buf.getvalue()).decode("utf-8").replace("\n", "")

        # Request response
        response = {
            "id": str(uuid.uuid4()),
            "chanceOfPassing": chance_of_passing,
            "shap-img": shap_img,
            "errors": errors
        }
    else:
        # Return errors
        response = {"id": str(uuid.uuid4()), "errors": errors}

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5045, debug=False)
