from flask import Flask, render_template, request
import pickle  # or import your model loading logic
import numpy as np

app = Flask(__name__)

# Load your trained machine learning model
with open('my_model.pkl', 'rb') as model_file:  # Adjust the file name accordingly
    model = pickle.load(model_file)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get data from the form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        # Add more features as required

        # Prepare the input for the model
        input_data = np.array([[feature1, feature2]])  # Adjust based on your model's input shape

        # Make prediction
        prediction = model.predict(input_data)

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
