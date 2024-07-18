from flask import Flask, request, jsonify
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from io import BytesIO

# Initialize the Flask app
app = Flask(__name__)
print('Server Started')


# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using '+ device)


# Define local paths for saving models
processor_path = 'processor/trocr-large-printed'
model_path = 'models/trocr-large-printed'

# Check if the processor is already downloaded
if not os.path.exists(processor_path):
    print(f"Downloading and saving processor to {processor_path}")
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
    processor.save_pretrained(processor_path)
else:
    print(f"Loading processor from {processor_path}")
    processor = TrOCRProcessor.from_pretrained(processor_path)

# Check if the model is already downloaded
if not os.path.exists(model_path):
    print(f"Downloading and saving model to {model_path}")
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
    model.save_pretrained(model_path)
else:
    print(f"Loading model from {model_path}")
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)



# # Load models
# processor = TrOCRProcessor.from_pretrained('processor/trocr-large-printed')
# model = VisionEncoderDecoderModel.from_pretrained('models/trocr-large-printed').to(device)
# yolo_model = YOLO('models\wine44epoch_v8s.pt').to(device)


print('Ready to Process. Intialization Complete..')

def get_text_from_bounding_boxes(image, boxes, class_names, device):
    texts = []
    
    for box, class_name in zip(boxes, class_names):
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2] # roi is the cropped image for the labelled area

        # Convert processed ROI to a PIL image
        pil_image = Image.fromarray(roi)

        # OCR using TrOCR
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        label = f"{class_name}: {text.strip()}"
        texts.append(label)
        
        # Draw bounding box and label on the image(only if we want annotated image)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return texts, image

@app.route('/wine', methods=['POST'])
def predict():
    if 'image' not in request.json:
        return jsonify({'error': 'No image data provided'}), 400

    # Decode base64 image data
    try:
        base64_image = request.json['image']
        image_data = base64.b64decode(base64_image)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({'error': f'Invalid image data: {str(e)}'}), 400

    if image is None:
        return jsonify({'error': 'Failed to decode image data'}), 400

    # Perform inference
    results = yolo_model(image)

    # Extract bounding boxes and class names
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding boxes
    class_ids = results[0].boxes.cls.cpu().numpy()  # Extract class IDs
    class_names = [yolo_model.names[int(class_id)] for class_id in class_ids]  # Get class names

    # Get text from bounding boxes
    texts, annotated_image = get_text_from_bounding_boxes(image, boxes, class_names, device)

    # Convert the annotated image to base64
    _, buffer = cv2.imencode('.jpg', annotated_image)
    annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Return the results
    return jsonify({
        'data': texts,
        # 'annotated_image': annotated_image_base64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
