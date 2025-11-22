import os
from flask import Flask, request, jsonify
from google import genai
from google.genai.errors import APIError

app = Flask(__name__)

# --- AI CLIENT INITIALIZATION ---
# The client automatically reads the GEMINI_API_KEY from the environment
# variables set in your deployment platform (GCP Cloud Run).
try:
    client = genai.Client()
    print("Gemini client initialized successfully.")
except Exception as e:
    # This will catch errors if the API key is missing or invalid on startup
    print(f"FATAL ERROR: Could not initialize Gemini client: {e}")
    client = None
# -----------------------------------

@app.route('/')
def home():
    if client:
        return jsonify({"status": "Online", "message": "Gemini AI Server is ready!"})
    else:
        return jsonify({"status": "Error", "message": "AI Client failed to initialize. Check API Key."}), 500

@app.route('/generate', methods=['POST'])
def generate_image():
    # Check if the client is available before processing
    if not client:
        return jsonify({"status": "error", "message": "AI Client not initialized."}), 500
        
    # 1. Get the text prompt from the Android App
    data = request.json
    prompt = data.get('prompt', 'A detailed, photorealistic cyborg cat painting a picture')
    
    print(f"Received prompt: {prompt}")
    
    try:
        # 1. New Model Call: Use generate_content instead of generate_images 
        #    and the model name 'gemini-2.5-flash-image'.
        result = client.models.generate_content(
            model='gemini-2.5-flash-image', 
            contents=[prompt],  # Pass the prompt as contents
            config=dict(
                response_modality="IMAGE", # Tell the model you want an image back
                image_config=dict(
                    aspect_ratio="1:1"
                )
            )
        )
        
        # 2. Extract the URL from the response structure
        if result.parts and result.parts[0].inline_data:
            # The URL is embedded in the response structure
            image_url = result.parts[0].inline_data.uri 
            
            return jsonify({
                "status": "success",
                "image_url": image_url
            })
        else:
            return jsonify({"status": "error", "message": "Image generation failed or model returned unexpected data."}), 500
            
    except APIError as e:
        # Handle specific API errors (e.g., prompt filtered, quota exceeded)
        print(f"Gemini API Error: {e.message}")
        return jsonify({"status": "error", "message": f"Gemini API Error: {e.message}"}), 500
    
    except Exception as e:
        print(f"Internal Server Error: {str(e)}")
        return jsonify({"status": "error", "message": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # This block is only used for local testing, not on Cloud Run
    app.run(debug=True, port=5000, host='0.0.0.0')