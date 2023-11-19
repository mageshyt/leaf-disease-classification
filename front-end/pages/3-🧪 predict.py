import streamlit as st
import requests
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

app_url = "http://127.0.0.1:8000/api"
endpoint = "/predict"
model_url = f"{app_url}{endpoint}"
class_names=['Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']


def read_image( image_encode):
    pil_image = Image.open(image_encode)
    return pil_image


def display_prediction(image, pred_label, pred_prob):
    """
    Display the image along with the predicted label and probabilities.
    """
    image=read_image(image)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the image (assuming image is a valid PIL Image)
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title(f'Prediction: {pred_label}')

    pred_prob=np.array(pred_prob)
    # Display the top 5 prediction probabilities as a bar chart
    top5_indices = np.argsort(pred_prob)[::-1][:5]
    top5_labels = [class_names[i] for i in top5_indices]
    top5_probs = pred_prob[top5_indices.astype(int)]  # Ensure indices are integers
    ax2.barh(np.arange(5), top5_probs, color='purple')
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(5))
    ax2.set_yticklabels(top5_labels, size='small')
    ax2.set_xlim(0, 1.1)
    ax2.set_xlabel('Prediction Probability')
    ax2.set_title('Top 5 Predictions')

    plt.tight_layout()

    return fig
    
st.set_page_config(page_title="Prediction",layout="centered")

st.title("Leaf Disease Prediction 🍀")

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Leaf Disease Prediction")
    st.info("This is a demo application written to show how to our deeplearing model can be used to predict the disease of a leaf.")



st.markdown("#### Upload an image of a leaf 🍃")
# upload image

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

cambera = st.checkbox("Use Camera")

if cambera:
    st.title("Camera")
    uploaded_file=st.camera_input(label="open camera")


if uploaded_file is not None:
    st.toast("Image uploaded successfully")

    # display image
    st.image(uploaded_file, width=300, caption="Uploaded Image")

    # get prediction

    if st.button("Predict" ,use_container_width=True):

        # prepare image for prediction
        files = {"file": uploaded_file.getvalue()}
        # make prediction
        response = requests.post(model_url, files=files)
        # display prediction
        pred = response.json()
        st.success(f"Prediction: {pred['prediction']}, Probability: {pred['probability']:.3f}")

        st.write("Top 5 Predictions")

        result=display_prediction(uploaded_file,pred['prediction'],pred['pred_prob'])
        st.pyplot(result)
        st.balloons()