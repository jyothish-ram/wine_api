from flask import Flask, request, jsonify
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, TRANSFORMERS_CACHE
from PIL import Image
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os

# Initialize the Flask app
app = Flask(__name__)
print('Server Started')

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using ' + device)

# # Define local paths for saving models
# processor_path = 'processor/trocr-large-printed'
# model_path = 'models/trocr-large-printed'

# # Check if the processor is already downloaded
# if not os.path.exists(processor_path):
#     print(f"Downloading and saving processor to {processor_path}")
#     processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
#     processor.save_pretrained(processor_path)
# else:
#     print(f"Loading processor from {processor_path}")
#     processor = TrOCRProcessor.from_pretrained(processor_path)

# # Check if the model is already downloaded
# if not os.path.exists(model_path):
#     print(f"Downloading and saving model to {model_path}")
#     model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
#     model.save_pretrained(model_path)
# else:
#     print(f"Loading model from {model_path}")
#     model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)



yolo_model = YOLO('models/wine44epoch_v8s.pt').to(device)
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed').to(device)
print('path for model & processor' + TRANSFORMERS_CACHE)

print('Ready to Process. Initialization Complete.')

def get_text_from_bounding_boxes(image, boxes, class_names, confidences, device):
    texts = {}
    for box, class_name, confidence in zip(boxes, class_names, confidences):
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]  # ROI is the cropped image for the labeled area

        # Convert processed ROI to a PIL image
        pil_image = Image.fromarray(roi)

        # OCR using TrOCR
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # If the class_name is not in texts or if the new confidence is higher, update the text
        if class_name not in texts or confidence > texts[class_name]['confidence']:
            texts[class_name] = {'text': text.strip(), 'confidence': float(confidence)}

        # Draw bounding box and label on the image (optional)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{class_name}: {text.strip()} ({confidence:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return texts, image

@app.route('/wine', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        # Decode base64 image data
        base64_image = request.json['image']
        image_data = base64.b64decode(base64_image)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Invalid image data: {str(e)}'}), 400

    if image is None:
        return jsonify({'error': 'Failed to decode image data'}), 400

    try:
        # Perform inference
        results = yolo_model(image)

        # Extract bounding boxes, class names, and confidences
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding boxes
        class_ids = results[0].boxes.cls.cpu().numpy()  # Extract class IDs
        confidences = results[0].boxes.conf.cpu().numpy()  # Extract confidence scores

        # Ensure class_ids are integers
        class_ids = class_ids.astype(int)

        # Get class names
        class_names = [yolo_model.names[class_id] for class_id in class_ids]

        # Get text from bounding boxes
        texts, annotated_image = get_text_from_bounding_boxes(image, boxes, class_names, confidences, device)

        # Initialize dictionary with all specified class names set to null
        specified_class_names = [
            "AlcoholPercentage", "AppellationQualityLevel", "AppellationRegion", "Country",
            "DistinctLogo", "EstablishedYear", "MakerName", "VintageYear", "WineType",
            "Sweetness", "Organic", "Sustainable"
        ]
        combined_texts = {class_name: None for class_name in specified_class_names}

        # Update dictionary with detected class names and their corresponding values
        for class_name, details in texts.items():
            combined_texts[class_name] = details['text']

        # Convert the annotated image to base64 (optional)
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Return the results
        return jsonify({
            'data': combined_texts,
            # 'annotated_image': annotated_image_base64  # Optional if you want to return the annotated image
        })
    except Exception as e:
        return jsonify({'error': f'Failed to process image: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
