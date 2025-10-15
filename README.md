# 🌾 **KissanAI — Revolutionizing Agriculture with Artificial Intelligence**  
### *Empowering India’s Farmers through Multilingual, Voice-Enabled, and Data-Driven Solutions*

Agriculture remains the backbone of India’s economy, yet millions of farmers continue to struggle with unpredictable weather, limited access to expert guidance, and complex agricultural data locked behind language barriers.

**KissanAI** is an intelligent, AI-powered digital assistant that transforms traditional farming into **smart, data-driven, and inclusive agriculture**.  
It acts as a **one-stop solution for every farmer’s need** — from **crop selection to yield prediction, disease detection, and market price forecasting** — all delivered through an easy-to-use **voice and language-adaptive web interface**.

---

## 🚜 **Problem Description**

Indian farmers face numerous challenges:
- Unpredictable **weather patterns** and **climate changes**
- **Language barriers** preventing easy access to agricultural insights
- Lack of **personalized, data-driven crop recommendations**
- Unawareness of **market price fluctuations**
- No real-time **disease or pest detection** system

**KissanAI** addresses these issues by combining **AI, voice recognition, and multilingual accessibility** to deliver actionable, real-time insights — ensuring that **every farmer, regardless of region or literacy, can make informed farming decisions.**

---

## 🌱 **Key Features**

| Feature | Description |
|----------|--------------|
| 🗣️ **Voice-Enabled Input** | Farmers can communicate using voice in their **native languages** like Hindi, Telugu, Tamil, Bengali, etc. |
| 🌾 **Crop Recommendation** | Suggests the best crop based on soil nutrients, weather, pH, and rainfall using ML models. |
| 🌱 **Crop Yield Prediction** | Predicts expected yield (hg/ha) using weather, area, pesticide use, and soil conditions. |
| 💰 **Market Price Forecasting** | Predicts commodity prices based on location, variety, and time of year. |
| 🔬 **Pest & Disease Detection** | Uses AI (CNN model) to detect diseases from crop images. |
| 🌦️ **Weather Insights** | Provides real-time weather alerts and forecasts for better planning. |
| 💬 **AI Chatbot** | Conversational AI assistant to answer farming-related queries. |
| 📊 **Trend Analytics Dashboard** | Visual display of live data, weather, and market patterns. |
| 🌓 **Responsive Modern UI** | Built using TailwindCSS & Bootstrap for an intuitive, mobile-friendly experience. |

---

## 💡 **Wow Factors (Hackathon Highlights)**

✨ **1. Multilingual & Voice AI** — Supports **12+ Indian languages** with natural speech input.  
🌤️ **2. Live Weather & Alerts** — Real-time weather and regional farming warnings.  
🤖 **3. AI Chatbot Integration** — Context-aware conversational assistant for farming help.  
📈 **4. Market Intelligence** — Predicts price trends across states and commodities.  
📱 **5. Scalable Architecture** — Flask-based modular ML model integration.  
🔗 **6. Transparent ML Pipeline** — All notebooks and trained models are publicly shared via Google Drive.  

---

## 🏗️ **System Architecture**




### 🏗️ System Architecture

```
🌾 User (Farmer)
   │
   ├── 🎤 Voice Input (Native Language)
   │       ↓
   ├── 🧠 NLP Chatbot & Translation Engine
   │       ↓
   ├── 🔍 AI Analysis (Flask Backend)
   │       ├── Crop Recommendation (Random Forest)
   │       ├── Crop Yield Prediction (Decision Tree)
   │       ├── Market Analysis (Random Forest)
   │       └── Pest Detection (CNN - optional)
   │
   └── 💻 Web Dashboard (Tailwind, JS, Flask)
            ↓
       🌦️ Weather, 📊 Analytics, 💬 Chatbot
```


> 🧠 *Note:* The NLP chatbot and translation engine are under development.  
> The current version uses **frontend speech recognition** and a **Flask-based logic layer** for processing.  
> Future updates will integrate **Hugging Face NLP models** and **Google Translate API**.

---

## 📚 **Model Development Process**

📁 **Models & Notebooks:**  
[🔗 Google Drive — All Models & Notebooks](https://drive.google.com/drive/folders/1VHVGu8IsYvnE87f3ZZ6DBHdUvQr7HzR4?usp=drive_link)

### 1️⃣ Crop Recommendation
- **Algorithm:** Random Forest Classifier  
- **Input Features:** N, P, K, Temperature, Humidity, pH, Rainfall  
- **Dataset:** Kaggle “Crop Recommendation Dataset”  
- **Output:** Recommended Crop  
- **Accuracy:** ~98%  
- **Exported Files:** `model.pkl`, `minmaxscaler.pkl`

### 2️⃣ Crop Yield Prediction
- **Algorithm:** Decision Tree Regressor  
- **Input:** Year, Rainfall, Temperature, Pesticides, Area, Crop Item  
- **Output:** Predicted Yield (hg/ha)  
- **Exported Files:** `dtr.pkl`, `preprocesser.pkl`

### 3️⃣ Market Price Prediction
- **Algorithm:** Random Forest Regressor  
- **Input:** State, District, Commodity, Variety, Month, Year  
- **Output:** Predicted Market Price (₹/Quintal)  
- **Exported Files:** `crop_price_model.pkl`, `le_state.pkl`, `le_district.pkl`, `le_commodity.pkl`, `le_variety.pkl`

### 4️⃣ Pest Detection (Optional Module)
- **Algorithm:** Convolutional Neural Network (CNN)  
- **Dataset:** PlantVillage  
- **Output:** Detected Disease Name  
- **Framework:** TensorFlow/Keras  

---

## 🗂️ **Project Folder Structure**

### 🗂️ Folder Structure

```
KisanAI/
├── mainapp.py
│
├── models/
│   ├── crop recommendation/
│   │   ├── model.pkl
│   │   └── minmaxscaler.pkl
│   │
│   ├── crop yield/
│   │   ├── dtr.pkl
│   │   └── preprocesser.pkl
│   │
│   ├── market analysis/
│   │   ├── crop_price_model.pkl
│   │   ├── le_state.pkl
│   │   ├── le_district.pkl
│   │   ├── le_commodity.pkl
│   │   └── le_variety.pkl
│   │
│   └── pest_detection/ (optional)
│       └── plant_disease_model.keras
│
├── templates/
│   ├── practice.html              # Main Home Page (Landing UI)
│   ├── crop_recommendation.html
│   ├── crop_yield.html
│   ├── market_main.html
│   ├── market_result.html
│   ├── pest_detection.html
│   └── some.html
│
├── crop_static/
│   ├── rice.jpg
│   ├── maize.jpg
│   └── ... (other crop images)
│
├── uploadimages/                  # Stores pest upload images
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/<your-username>/KissanAI.git
cd KissanAI
```

### Steps 2, 3 & 4: Setup Environment, Install Dependencies & Run App

**For Linux/macOS**
```bash
# Step 2: Create virtual environment
python -m venv venv
source venv/bin/activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run Flask app
python mainapp.py
```

**For Windows**
```bat
:: Step 2: Create virtual environment
python -m venv venv
venv\Scripts\activate

:: Step 3: Install dependencies
pip install -r requirements.txt

:: Step 4: Run Flask app
python mainapp.py
```

Open your browser at 👉 [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---
## 📜 requirements.txt
```text
Flask
numpy
pandas
scikit-learn
joblib
pickle-mixin
tensorflow
SpeechRecognition
pyttsx3
gunicorn
```

---

## 💬 Chatbot & Voice Assistant

- Built using **SpeechRecognition API** and **Web Speech API**  
- Supports **text + voice queries**  
- Detects and responds in **local languages**  
- Future-ready for **NLP integration** using Transformers / LLMs  

### Assists in:
- Crop problem diagnosis  
- Fertilizer recommendations  
- Weather and market insights  

---

## ☁️ Deployment

Can be deployed easily on:

- Render  
- Railway  
- Heroku  

**Procfile Example:**
```text
web: gunicorn mainapp:app
```

Add environment variable:
```text
PEST_DETECTION_ENABLED=False
```

---

## 👨‍🌾 Team KissanAI

| Member           | Role                   | Responsibilities                               |
|------------------|-----------------------|------------------------------------------------|
| Mahammad Usman   | ML Model Developer                | Model development, Flask backend integration |
| Naru surya charan| Frontend/UI Developer             | Tailwind UI, chatbot integration             |
| Mohammad sajid   | Back-End Developer                | Data preprocessing, cleaning, model evaluation|
| Ibrahim          | Documentation and presentation lead|Report writing and hackathon presentation    |
| Ilakkiya         | UI/UX Designer                   | Making rich interfaces                         |

---

## 🚀 Future Enhancements

- 🌾 Real-time soil health sensors (IoT)  
- 📱 Android app with offline access  
- 🧠 NLP chatbot integration with translation APIs  
- 🛰️ Satellite-based disease detection  
- 💡 Personalized fertilizer recommendation engine  

---

## 🏁 Conclusion

KissanAI is not just a project — it’s a vision to make Indian agriculture **intelligent, inclusive, and accessible** for every farmer.  

By combining AI, multilingual voice support, and real-time data, we help farmers make **better, faster, and more profitable decisions**.  

🌾 From soil to market — **KissanAI guides every step**.





