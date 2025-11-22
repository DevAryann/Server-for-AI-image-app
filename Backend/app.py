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
        # 2. Call the Image Generation Model (Imagen 3)
        # Using a fixed aspect ratio (1:1) is often best for standard UI cards
        result = client.models.generate_images(
            model='imagen-3.0-generate-002', 
            prompt=prompt,
            config=dict(
                number_of_images=1,
                output_mime_type="image/jpeg",
                aspect_ratio="1:1"
            )
        )
        
        # 3. Process the result and get the URL
        if result.generated_images:
            # The .uri attribute provides the public URL of the generated image
            image_url = result.generated_images[0].uri
            
            return jsonify({
                "status": "success",
                "image_url": image_url
            })
        else:
            return jsonify({"status": "error", "message": "Image generation failed or returned no images."}), 500
            
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