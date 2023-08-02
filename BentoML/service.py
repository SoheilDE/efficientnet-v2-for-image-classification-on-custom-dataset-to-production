import cv2
import bentoml
from bentoml.io import Image
from bentoml.io import Text
import numpy as np
from PIL.Image import Image as PILImage

runner = bentoml.keras.get('efficientnet_model:latest').to_runner()

svc = bentoml.Service("efficientnet_model", runners=[runner])


@svc.api(input=Image(), output=Text())
async def predict_image(image: PILImage) -> str:

    img = np.array(image)
    img = cv2.imencode('.jpg', img)[1].tobytes()
    img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (384, 384))

    img = np.expand_dims(img, axis=0)
    #img = preprocess_input(img)

    prediction = await runner.async_run(img)
    predicted_classes = np.argmax(prediction, axis=1).astype(int)
    if prediction[0][predicted_classes[0]] < 0.7:
        predicted_classes[0] = 1
    class_labels = ['class1', 'class2']  # List of your class labels
    predicted_label = class_labels[predicted_classes[0]]

    return predicted_label
