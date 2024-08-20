from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
# Load the model and label encoders
model = joblib.load('best_xgb_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define the feature mapping
feature_mapping = {
    'age': 'Age',
    'gender': 'Gender',
    'place_of_residence': 'Place of recidence',
    'occupation': 'Occupation',
    'in_store_payments': 'Making in-store payments(at shops and restaurants)',
    'paying_bills': 'Paying bills(utilities, top-ups)',
    'transferring_money': 'Transferring money to another user',
    'ease_of_navigation': 'Ease of navigation',
    'visually_appealing': 'Visually appealing and user-friendly',
    'ease_of_finding_features': 'Ease of finding features and functions',
    'security_features': 'Security features',
    'comfort_of_collecting_info': 'comfortability of collecting financial information',
    'security_issues_present': 'security issues present',
    'bill_payments': 'Bill payments',
    'money_transfers': 'Money transfers between users',
    'top_up_capabilities': 'Top-up capabilities(phone, internet and gift cards)',
    'promotions_discounts': 'Promotions and discounts',
    'loyalty_programs': 'Loyalty programs and rewards',
    'missing_features': 'missing features',
    'reliability': 'overall reliability (uptime and performance)',
    'speed_of_transactions': 'speed of transactions',
    'customer_support': 'customer support',
    'contact_customer_support_for_any_issues': 'contact customer support for any issues',
    'recommend_the_mobile_payment_applications_you_use_to_others': 'recommend the mobile payment applications you use to others'
}

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.json
        
        # Transform the input data using the feature mapping
        transformed_data = {feature_mapping[key]: value for key, value in data.items()}
        
        # Convert transformed data to DataFrame
        df = pd.DataFrame([transformed_data])
        
        # Encode categorical features using the loaded label encoders
        for column in df.columns:
            if column in label_encoders:
                le = label_encoders[column]
                try:
                    df[column] = le.transform(df[column])
                except ValueError:
                    # Handle unseen labels by assigning a default value
                    unique_classes = le.classes_.tolist()
                    if np.issubdtype(df[column].dtype, np.number):
                        default_value = min(unique_classes)
                    else:
                        default_value = unique_classes[0]
                    df[column] = df[column].apply(lambda x: le.transform([x])[0] if x in unique_classes else default_value)
        
        # Ensure all columns are of the correct type
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].astype('category').cat.codes
        
        # Predict using the model
        prediction = model.predict(df)
        
        # Decode the prediction to original label
        prediction_label = label_encoders['Mobile Payment Application'].inverse_transform(prediction)[0]
        
        return jsonify({'Mobile Payment Application': prediction_label})
    except KeyError as e:
        return jsonify({'error': f'Missing key in input data: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
