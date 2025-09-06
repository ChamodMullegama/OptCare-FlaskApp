from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import re
import google.generativeai as genai
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["5 per minute"]
)

# Configure Gemini API key
GEMINI_API_KEY = 'AIzaSyAUmgzUTzDLXV8TyT6DsOFr1hBZod_jqe4'  
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Load OCT classification model
oct_model = tf.keras.models.load_model("./models/final_retinal_disease_model.keras")
labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Load OCT detector model (add this)
oct_detector_model = tf.keras.models.load_model("./models/oct_detector_model.h5")  

def is_oct_image(image_file, threshold=0.5):
    try:
        image = Image.open(image_file).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  
        
        # Make prediction
        prediction = oct_detector_model.predict(img_array)[0][0]
        is_oct = prediction > threshold
        
        return is_oct, float(prediction)
    except Exception as e:
        print(f"Error in OCT detection: {str(e)}")
        return False, 0.0

def model_prediction(test_image):
    """Process image and return prediction for OCT classification"""
    image = Image.open(test_image).convert("RGB")
    image = image.resize((224, 224))
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    input_arr = tf.keras.applications.mobilenet_v3.preprocess_input(input_arr)
    predictions = oct_model.predict(input_arr)
    return np.argmax(predictions)

def get_gemini_recommendation(prediction):
    """Get AI-generated recommendations from Gemini in English"""
    prompt = f"""
    As a senior ophthalmologist, provide a detailed clinical recommendation for a patient diagnosed with {prediction} based on OCT scan findings. Provide the response in English only, formatted in clean HTML with <h4> for section headers and <ul> for lists. Structure the response as follows:

    **Clinical Summary**: Brief explanation of {prediction} in 2-3 sentences

    **Recommended Interventions**:
    - First-line treatments (medications, procedures)
    - Alternative options if first-line fails
    - Emerging therapies (if applicable)

    **Monitoring Plan**:
    - Recommended follow-up schedule
    - Key parameters to monitor
    - Imaging frequency

    **Patient Counseling Points**:
    - Symptoms requiring immediate attention
    - Lifestyle modifications
    - Visual aids if needed

    Format the response as:
    <div class='english-recommendation'>
      <h3>Clinical Recommendation</h3>
      <!-- English content with h4 and ul -->
    </div>
    """

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2000
            )
        )
        if not response.text:
            raise ValueError("Empty response from Gemini API")
        
        cleaned_response = re.sub(r'```html\s*|\s*```', '', response.text).strip()
        return cleaned_response
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return """
        <div class='english-recommendation'>
            <h3>Clinical Recommendation</h3>
            <p>Unable to generate recommendations due to an API error. Please consult an ophthalmologist for detailed advice.</p>
        </div>
        """

@app.route('/predict', methods=['POST'])
@limiter.limit("5 per minute")
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    try:
        image_file = request.files['image']
        
        if not image_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({'error': 'Invalid file format'}), 400
        
        is_oct, confidence = is_oct_image(image_file)
        
        if not is_oct:
            return jsonify({
                'error': f'The provided image does not appear to be an OCT scan (confidence: {confidence:.2%}). Please upload a valid OCT scan image.'
            }), 400
            
        # Reset file pointer for the next model
        image_file.seek(0)
        
        # If it's an OCT scan, proceed with classification
        result_index = model_prediction(image_file)
        predicted_class = labels[result_index]
        recommendation = get_gemini_recommendation(predicted_class)
        
        return jsonify({
            'prediction': predicted_class,
            'recommendation': recommendation,
            'is_oct_confidence': confidence
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Processing failed: {str(e)}'
        }), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)