#Load libraries
import flask
import numpy as np
from keras.models import load_model

#Instantiate flas
app = flask.Flask(__name__)

model = load_model('./../model.h5')

@app.route('/api/predict')
def predict():
    test_x = np.array([[22.93],[20.53],[21.53],[23.23],[23.03],[24.48],[24.78],[27.22],[27.63],[26.1 ],[21.76],[24.1 ],[24.9 ],[28.92],[28.27],[26.97],[25.52]])
    test_x = test_x.reshape(1,17,1)
    pred_y = model.predict(test_x)[0].tolist()

    response = dict()
    response['input'] = test_x[0].flatten().tolist()
    response['prediction'] = pred_y
    return str(response)

app.run(host='0.0.0.0')
