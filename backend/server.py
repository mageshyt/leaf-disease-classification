from fastapi import FastAPI,File
import tensorflow as tf
import tensorflow_hub as hub
from utils import ImagePredictor
import base64

# Create a FastAPI instance
app = FastAPI(
    title="Leaf Disease Prediction",
    description="This is a demo application written to show how to our deeplearing model can be used to predict the disease of a leaf.",
    version="0.1",
)


# Define a root `/` endpoint

@app.get("/")

def index():
    return {"message": "Hello World"}



# Create a function to load a trained model
def load_model(model_path):
  """
  Loads a saved model from a specified path.
  """
  print(f"Loading saved model from: {model_path}")
  model = tf.keras.models.load_model(model_path,
                                     custom_objects={"KerasLayer":hub.KerasLayer})
  return model


model = load_model('../models/20231008-22141696783457-21k-images-mobilenetv2-Adam.h5')
class_names=['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
predict=ImagePredictor(model=model, class_names=class_names)

@app.post('/api/predict')

async def getPrediction(file: bytes = File(...)):
    """
    Predicts the class label for the given image and displays the result.
    """

    # Get the predicted label
    result = predict.predict(file)

    pred_label = result['prediction']

    max_prob = result['max-probability']

    pred_prob = result['probabilities']
    # Display the image and prediction
    return {"prediction": pred_label, "probability": max_prob,"pred_prob": pred_prob}
