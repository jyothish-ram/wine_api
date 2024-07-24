#This is only for web ui upload 
# only for debugging


from flask import Flask, request, jsonify, render_template, send_from_directory
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, TRANSFORMERS_CACHE
import cv2
import pytesseract
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os
import torch
import base64
from io import BytesIO


# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using '+ device)



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



yolo_model = YOLO('models\wine44epoch_v8s.pt').to(device)
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed').to(device)
print('path for model & processor' + TRANSFORMERS_CACHE)



print('Ready to Process. Intialization Complete..')

app = Flask(__name__)
model = YOLO('models/wine44epoch_v8s.pt')  # Load your model


def get_text_from_bounding_boxes(image, boxes, class_names):
    texts = []
    for box, class_name in zip(boxes, class_names):
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]

        # Convert processed ROI to a PIL image
        pil_image = Image.fromarray(roi)

        # OCR using TrOCR
        pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        label = f"{class_name}: {text.strip()}"
        texts.append(label)

        
        # Draw bounding box and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return texts, image

@app.route('/wine_app', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            upload_folder = 'static/uploads'
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            image = cv2.imread(file_path)
            results = model(image)  # Perform inference

            boxes = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding boxes
            class_ids = results[0].boxes.cls.cpu().numpy()  # Extract class IDs
            class_names = [model.names[int(class_id)] for class_id in class_ids]  # Get class names

            texts, annotated_image = get_text_from_bounding_boxes(image, boxes, class_names)

            # Save the annotated image
            annotated_image_filename = 'annotated_' + file.filename
            annotated_image_path = os.path.join(upload_folder, annotated_image_filename)
            cv2.imwrite(annotated_image_path, annotated_image)

            return render_template('result.html', 
                                   texts=texts, 
                                   annotated_image_filename=annotated_image_filename)

    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static/uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)
