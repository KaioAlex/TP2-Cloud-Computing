#import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "<p>Hello, World!</p>"

'''
@app.route('/predict', methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))
'''

@app.route('/api/recommender', methods=['POST'])
def recommend():
    data = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])

    #output = prediction[0]
    #return jsonify(output)

    print(data)
    return data

if __name__ == "__main__":
    app.run(host="localhost", port=32174, debug=True)