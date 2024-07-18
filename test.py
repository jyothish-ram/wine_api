import requests
import base64

# Function to encode an image file to base64 string
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Example usage: Sending base64 encoded image to Flask API
def send_image_to_api(image_path):
    api_url = 'http://127.0.0.1:5000/wine'
    image_base64 = encode_image_to_base64(image_path)
    data = {
        'image': image_base64
    }
    headers = {
        'Content-Type': 'application/json'
    }
    response = requests.post(api_url, json=data, headers=headers)
    print(response.status_code)
    if response.status_code == 200:
        print(response.json())
    else:
        print("Error:", response.text)

# Example usage
send_image_to_api('static/uploads/wine_tough.webp')
