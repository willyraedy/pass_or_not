import flask
import pickle
import numpy as np
import pandas as pd

#---------- MODEL IN MEMORY ----------------#

with open('./finalModel.pickle', 'rb') as read_file:
    PREDICTOR = pickle.load(read_file)


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, awesome.html
    """
    with open("index.html", 'r') as viz_file:
        return viz_file.read()

# Get an example and return it's score from the predictor model
@app.route("/score", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    data = flask.request.json
    inputs = pd.DataFrame([dict(data)])
    score = PREDICTOR.predict_proba(inputs)
    # Put the result in a nice dict so we can send it as json
    results = {"result": 1 if score[:, 1][0] > 0.75 else 0, 'score': score[:, 1][0] }
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0', port=89)
app.run(debug=True)
