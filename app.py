import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained machine learning model
model = None

# Load your training data and train the model
def load_model_and_data():
    global model
    if model is None:
        model = joblib.load('Travel.pkl')

@app.before_request
def before_request():
    load_model_and_data()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    Age = int(request.form['Age'])
    EmploymentType = request.form['EmploymentType']
    GraduateOrNot = request.form['GraduateOrNot']
    AnnualIncome = int(request.form['AnnualIncome'])
    FamilyMembers = int(request.form['FamilyMembers'])
    ChronicDiseases = request.form['ChronicDiseases']
    FrequentFlyer = request.form['FrequentFlyer']
    EverTravelledAbroad = request.form['EverTravelledAbroad']
    print(FrequentFlyer,EverTravelledAbroad,ChronicDiseases,EmploymentType)
    # Encode categorical variables
    # EmploymentType_encoded = 1 if EmploymentType == 'Private Sector/Self Employed' else 0
    # FrequentFlyer_encoded = 1 if FrequentFlyer == 'Yes' else 0
    # EverTravelledAbroad_encoded = 1 if EverTravelledAbroad == 'Yes' else 0

    # Create a list of features in the same order as the model expects
    features = [Age, EmploymentType, GraduateOrNot, AnnualIncome, FamilyMembers, ChronicDiseases, FrequentFlyer, EverTravelledAbroad]
    print(features)
    # Make a prediction using the loaded and trained model
    prediction = model.predict(features)
    if prediction == 1:
        prediction_text = 'Yes'
    else:
        prediction_text = 'No'

    return render_template('index.html', prediction_text=prediction_text)


if __name__ == "__main__":
    app.run(debug=True)
