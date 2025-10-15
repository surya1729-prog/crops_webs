import os
#import uuid # Re-added for future use with pest detection
import numpy as np
import pickle
import joblib
#import tensorflow as tf # Re-added for future use with pest detection
from flask import Flask, request, render_template, url_for #, redirect, send_from_directory # Re-added redirect and send_from_directory

# ==============================================================================
# 1. INITIALIZE FLASK APP & CONFIGURE
# ==============================================================================
# Your HTML files must be in a 'templates' folder.
# Your static images (e.g., rice.jpg) must be in a 'crop_static' folder.
app = Flask(__name__, static_folder='crop_static',static_url_path='/static')
# Create a directory to store uploaded images for pest detection
UPLOAD_FOLDER = 'uploadimages'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- FEATURE FLAG TO DISABLE PEST DETECTION ---
PEST_DETECTION_ENABLED = False # Set to False to disable, True to enable

# ==============================================================================
# 2. LOAD ALL MACHINE LEARNING MODELS & SCALERS
# ==============================================================================
# --- Crop Recommendation Model ---
try:
    # CORRECTED: Standardized spelling to "recommendation"
    crop_rec_model = pickle.load(open(os.path.join("models","crop recommendation","model.pkl"), "rb"))
    crop_rec_scaler = pickle.load(open(os.path.join("models","crop recommendation","minmaxscaler.pkl"), "rb"))
except FileNotFoundError:
    print("Crop recommendation model/scaler not found. Ensure the folder is named 'crop recommendation'.")
    crop_rec_model, crop_rec_scaler = None, None

# --- Crop Yield Model ---
try:
    crop_yield_model = pickle.load(open(os.path.join("models","crop yield","dtr.pkl"), "rb"))
    crop_yield_preprocessor = pickle.load(open(os.path.join("models","crop yield","preprocesser.pkl"), "rb"))
except FileNotFoundError:
    print("Crop yield model/preprocessor not found.")
    crop_yield_model, crop_yield_preprocessor = None, None

# --- Market Analysis Model ---
try:
    market_model = joblib.load(os.path.join("models","market analysis", "crop_price_model.pkl"))
    le_state = joblib.load(os.path.join("models","market analysis", "le_state.pkl"))
    le_district = joblib.load(os.path.join("models","market analysis", "le_district.pkl"))
    le_commodity = joblib.load(os.path.join("models","market analysis", "le_commodity.pkl"))
    le_variety = joblib.load(os.path.join("models","market analysis", "le_variety.pkl"))
except FileNotFoundError:
    print("Market analysis model/encoders not found.")
    market_model = le_state = le_district = le_commodity = le_variety = None

# --- Pest Detection Model (Commented Out for future use) ---
"""
if PEST_DETECTION_ENABLED:
    try:
        # Disable GPU for consistency
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        pest_model = tf.keras.models.load_model(os.path.join("models", "pest_detection", "plant_disease_model.keras"))
    except (FileNotFoundError, IOError):
        print("Pest detection model not found.")
        pest_model = None
else:
    pest_model = None
    print("Pest detection is DISABLED by configuration.")
"""

# ==============================================================================
# 3. MAPPINGS & HELPER FUNCTIONS
# ==============================================================================
# --- Crop Recommendation Mappings ---
crop_rec_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}
crop_rec_images = {crop: f"{crop.lower()}.jpg" for crop in crop_rec_dict.values()}

# --- Pest Detection Mappings (Commented Out for future use) ---
"""
pest_labels = [
 'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
 'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
 'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]
"""

# --- Market Analysis Helper ---
def safe_transform(encoder, value):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    elif 'Unknown' in encoder.classes_:
        return encoder.transform(['Unknown'])[0]
    else:
        return -1

# ==============================================================================
# 4. FLASK ROUTES
# ==============================================================================
@app.route('/')
def home():
    """Renders the main homepage."""
    return render_template('practice.html')

# --- CROP RECOMMENDATION ROUTE ---
@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    result = None
    result_img = None
    if request.method == 'POST':
        try:
            # CORRECTED: "Phosphorus" spelling
            features = [float(request.form[key]) for key in ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Ph', 'Rainfall']]

            single_pred = np.array(features).reshape(1, -1)
            scaled_features = crop_rec_scaler.transform(single_pred)
            prediction = crop_rec_model.predict(scaled_features)
            crop_name = crop_rec_dict.get(prediction[0], "Unknown")
            result = f"🌾 {crop_name} is the best crop for your farm!"
            result_img = crop_rec_images.get(crop_name, "default_crop.jpg")
        except Exception as e:
            result = f"⚠️ Error processing your request: {e}"
    # CORRECTED: Renders the correctly named HTML file
    return render_template('crop_recommendation.html', result=result, result_img=result_img)

# --- CROP YIELD PREDICTION ROUTE ---
@app.route('/crop-yield', methods=['GET', 'POST'])
def crop_yield():
    prediction_text = None
    if request.method == 'POST':
        try:
            features = [
                request.form['Year'],
                request.form['average_rain_fall_mm_per_year'],
                request.form['pesticides_tonnes'],
                request.form['avg_temp'],
                request.form['Area'].strip().title(),
                request.form['Item'].strip().title()
            ]
            feature_array = np.array([features], dtype=object)
            transformed_features = crop_yield_preprocessor.transform(feature_array)
            prediction = crop_yield_model.predict(transformed_features).reshape(1, -1)
            prediction_text = f"Predicted Yield: {prediction[0][0]:.2f} hg/ha"
        except Exception as e:
            prediction_text = f"⚠️ Error predicting yield: {e}"
    
    return render_template('crop_yield.html', prediction=prediction_text)

# --- MARKET ANALYSIS ROUTE ---
@app.route('/market-analysis', methods=['GET', 'POST'])
def market_analysis():
    if request.method == 'POST':
        try:
            state = request.form['state'].strip().title()
            district = request.form['district'].strip().title()
            commodity = request.form['commodity'].strip().title()
            variety = request.form['variety'].strip().title()
            month = int(request.form['month'])
            year = int(request.form['year'])

            features = np.array([[
                safe_transform(le_state, state),
                safe_transform(le_district, district),
                safe_transform(le_commodity, commodity),
                safe_transform(le_variety, variety),
                month, year
            ]])
            
            price_pred = market_model.predict(features)[0]
            
            # IMPROVEMENT: Pass a dictionary with a success status for better error handling in HTML
            return render_template('market_result.html', result={
                "success": True,
                "price": f"{price_pred:.2f}"
            })

        except Exception as e:
            # IMPROVEMENT: If an error occurs, pass a success=False status
            return render_template('market_result.html', result={
                "success": False,
                "error": str(e)
            })

    # On GET request, just show the input form.
    return render_template('market_main.html')


# --- ADD THIS NEW ROUTE ---
# This tells Flask what to do when someone visits /pest_detection.html
@app.route('/pest_detection.html')
def pest_detection():
    # This line finds the file in your 'templates' folder and displays it.
    return render_template('pest_detection.html')

@app.route('/some.html')
def voice_input():
    # This line finds the file in your 'templates' folder and displays it.
    return render_template('some.html')

# --- PEST DETECTION ROUTES (Commented Out for future use) ---
"""
@app.route('/pest-detection', methods=['GET', 'POST'])
def pest_detection():
    # If the feature is disabled, show the unavailable page
    if not PEST_DETECTION_ENABLED:
        return render_template('pest_detection.html', unavailable=True)

    if request.method == 'POST':
        if 'img' not in request.files or request.files['img'].filename == '':
            return redirect(request.url)
        
        image = request.files['img']
        # Create a unique filename to avoid conflicts
        filename = f"{uuid.uuid4().hex}_{image.filename}"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        try:
            # Preprocess the image
            img = tf.keras.utils.load_img(image_path, target_size=(161, 161), color_mode='rgb')
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            feature = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = pest_model.predict(feature)
            predicted_label = pest_labels[np.argmax(prediction)]
            
            return render_template('pest_detection.html', result=True, imagepath=filename, prediction=predicted_label)
        except Exception as e:
            return render_template('pest_detection.html', error=f"⚠️ Error during prediction: {e}")

    return render_template('pest_detection.html')

@app.route('/uploads/<filename>')
def serve_uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
"""

# ==============================================================================
# 5. RUN THE FLASK APP
# ==============================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)

