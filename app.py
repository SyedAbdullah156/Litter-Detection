from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# LOAD TRAINED MODELS
try:
    real_estate_model = joblib.load('./models/real_estate/best_model.joblib')
    scaler = joblib.load('./models/real_estate/scaler.joblib')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load models. Error: {e}")
    real_estate_model = None
    scaler = None

def calculate_roi(current_price, predicted_price):
    """Calculate ROI percentage"""
    if current_price == 0:
        return 0
    roi = ((predicted_price - current_price) / current_price) * 100
    return round(roi, 2)

def get_recommendation(roi):
    """Get investment recommendation based on ROI"""
    if roi > 15:
        return {
            'status': 'STRONG BUY',
            'color': 'success',
            'icon': '🟢',
            'message': 'Excellent investment opportunity with high ROI!'
        }
    elif roi > 8:
        return {
            'status': 'MODERATE BUY',
            'color': 'info',
            'icon': '🟡',
            'message': 'Good investment with solid returns expected.'
        }
    elif roi > 3:
        return {
            'status': 'HOLD',
            'color': 'warning',
            'icon': '🟠',
            'message': 'Average returns. Consider market conditions.'
        }
    else:
        return {
            'status': 'CAUTION',
            'color': 'danger',
            'icon': '🔴',
            'message': 'Low returns expected. Explore alternatives.'
        }


# ROUTES

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predict_real_estate')
def predict_real_estate_page():
    """Real Estate Prediction Page"""
    return render_template('predict_real_estate.html')

@app.route('/api/predict_real_estate', methods=['POST'])
def predict_real_estate():
    """API endpoint for real estate prediction"""
    try:
        if real_estate_model is None or scaler is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please train the model first.'
            }), 500

        # Get form data
        data = request.json
        
        # Extract features (must match training features)
        avg_area_income = float(data.get('avg_area_income', 0))
        avg_area_house_age = float(data.get('avg_area_house_age', 0))
        avg_area_rooms = float(data.get('avg_area_rooms', 0))
        avg_area_bedrooms = float(data.get('avg_area_bedrooms', 0))
        area_population = float(data.get('area_population', 0))
        current_price = float(data.get('current_price', 0))
        
        # Create engineered features (same as training)
        income_per_room = avg_area_income / avg_area_rooms if avg_area_rooms > 0 else 0
        rooms_per_bedroom = avg_area_rooms / avg_area_bedrooms if avg_area_bedrooms > 0 else 0
        population_density = area_population / (avg_area_house_age + 1)
        
        # Prepare feature array (must match training order)
        features = np.array([[
            avg_area_income,
            avg_area_house_age,
            avg_area_rooms,
            avg_area_bedrooms,
            area_population,
            income_per_room,
            rooms_per_bedroom,
            population_density
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict
        predicted_price = real_estate_model.predict(features_scaled)[0]
        
        # Calculate ROI
        roi = calculate_roi(current_price, predicted_price)
        profit = predicted_price - current_price
        
        # Get recommendation
        recommendation = get_recommendation(roi)
        
        # Return results
        return jsonify({
            'success': True,
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'profit': round(profit, 2),
            'roi': roi,
            'recommendation': recommendation
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/compare')
def compare_page():
    """Investment Comparison Page (Future: Real Estate vs Stocks vs Gold)"""
    return render_template('compare.html')

@app.route('/about')
def about_page():
    """About page"""
    return render_template('about.html')

# RUN APP
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)