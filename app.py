from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define expected feature order
expected_features = ['Age', 'Gender', 'Education Level', 'Job Category Code', 'Years of Experience']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read form data
        age = int(request.form['Age'])
        gender = int(request.form['Gender'])  # 0=Female, 1=Male, etc.
        education = int(request.form['Education Level'])  # 0–3
        job_category = int(request.form['Job Category Code'])  # 0–3
        experience = float(request.form['Years of Experience'])

        # Create DataFrame in correct column order
        input_df = pd.DataFrame([[age, gender, education, job_category, experience]],
                                columns=expected_features)

        # Predict
        prediction = model.predict(input_df)[0]
        salary_output = f"${prediction:,.2f}"

        return render_template('index.html', prediction_text=f"Predicted Salary: {salary_output}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
