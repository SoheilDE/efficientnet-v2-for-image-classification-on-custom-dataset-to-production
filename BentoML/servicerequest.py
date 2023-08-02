import requests

url = 'http://127.0.0.1:3000/predict_image'

files = {
    'fileobj': ('image.jpeg', open('image.jpeg', 'rb'), 'image/jpeg')
}


response = requests.post(url, files=files)
print(response.text)
