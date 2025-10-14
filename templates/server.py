# 1. Install necessary libraries:
# pip install Flask pymongo flask-cors

from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import os

app = Flask(__name__)
CORS(app)  # This enables Cross-Origin Resource Sharing

# --- Database Connection ---
# The connection string is now correctly assigned to the MONGO_URI variable.
MONGO_URI = "mongodb+srv://usman:Usman771@cluster0.s3jxyep.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)

# Select your database (it will be created if it doesn't exist)
db = client['kissanai_db'] 
# Select your collection (it will be created if it doesn't exist)
collection = db['demo_requests']

@app.route('/')
def index():
    return "KissanAI Backend Server is running!"

# --- API Endpoint to receive form data ---
@app.route('/api/request-demo', methods=['POST'])
def request_demo():
    try:
        # Get data from the frontend request
        data = request.get_json()

        # Basic validation
        if not data or 'name' not in data or 'phone' not in data:
            return jsonify({"success": False, "error": "Missing required fields"}), 400

        # Insert the data into the MongoDB collection
        # The insert_one function returns an object with the inserted document's ID
        result = collection.insert_one(data)

        print(f"Successfully inserted data for {data['name']} with ID: {result.inserted_id}")

        # Send a success response back to the frontend
        return jsonify({"success": True, "message": "Demo request received!", "id": str(result.inserted_id)})

    except Exception as e:
        print(f"An error occurred: {e}")
        # Send an error response back to the frontend
        return jsonify({"success": False, "error": str(e)}), 500

# --- Running the server ---
if __name__ == '__main__':
    # The server will run on http://12.0.0.1:5000
    app.run(debug=True, port=5000)