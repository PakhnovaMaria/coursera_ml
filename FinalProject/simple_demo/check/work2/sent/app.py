import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.base import TransformerMixin
class DenseTransformer(TransformerMixin):
    def transform(self, X, y=None, **fit_params):
        return X.todense()
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
    def fit(self, X, y=None, **fit_params):
        return self
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features=list(request.form.values())
    prediction = model.predict(features)

    output = round(prediction[0], 2)

    kv={0:"good",1:"bad"}

    return render_template('index.html', prediction_text='This review is {}'.format(kv[output]))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    features=list(data.values())
    prediction = model.predict(features)

    output = {'result':prediction[0]}
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)
