import os
from flask import Flask, request, jsonify
from google import genai
from google.genai.errors import APIError
from google.genai import types # Needed for correctly configuring the API call

app = Flask(__name__)

# --- AI CLIENT INITIALIZATION ---
# The client automatically looks for the GEMINI_API_KEY environment variable.
try:
    # Attempt to initialize the client
    client = genai.Client()
    print("Gemini client initialized successfully.")
except Exception as e:
    # If the key is missing or invalid, the client will be None.
    print(f"FATAL ERROR: Could not initialize Gemini client: {e}")
    client = None
# -----------------------------------

@app.route('/')
def home():
    """Health check endpoint to ensure the server is running."""
    if client:
        return jsonify({"status": "Online", "message": "Gemini AI Server is ready!"})
    else:
        return jsonify({"status": "Error", "message": "AI Client failed to initialize. Check API Key."}), 500

@app.route('/generate', methods=['POST'])
def generate_image():
    """Handles the POST request for image generation."""
    
    # Check if the client is available
    if not client:
        return jsonify({"status": "error", "message": "AI Client not initialized."}), 500
        
    data = request.json
    # Use a high-quality default prompt in case the app doesn't send one
    prompt = data.get('prompt', 'A detailed, photorealistic vaporwave landscape with neon signs')
    
    print(f"Received prompt: {prompt}")
    
    try:
        # Define the Image Configuration (Aspect Ratio)
        image_config = types.ImageConfig(
            aspect_ratio="1:1"
        )
        
        # Define the entire request configuration, excluding the forbidden 'response_modality'
        config_object = types.GenerateContentConfig(
            image_config=image_config
        )

        # Call the Image Generation Model (Gemini 2.5 Flash Image)
        result = client.models.generate_content(
            model='gemini-2.5-flash-image', 
            contents=[prompt],
            config=config_object
        )
        
        # Extract the URL from the response structure
        if result.parts and result.parts[0].inline_data:
            # The .uri attribute provides the public URL of the generated image
            image_url = result.parts[0].inline_data.uri 
            
            return jsonify({
                "status": "success",
                "image_url": image_url
            })
        else:
            return jsonify({"status": "error", "message": "Image generation failed or model returned unexpected data."}), 500
            
    except APIError as e:
        # Catches API-specific errors (e.g., prompt filtering, quota exceeded)
        print(f"Gemini API Error: {e.message}")
        return jsonify({"status": "error", "message": f"Gemini API Error: {e.message}"}), 500
    except Exception as e:
        # Catches unexpected server errors (e.g., JSON parsing failure, internal server issue)
        print(f"Internal Server Error: {str(e)}")
        return jsonify({"status": "error", "message": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    # This is for local testing only (your laptop)
    app.run(debug=True, port=5000, host='0.0.0.0')