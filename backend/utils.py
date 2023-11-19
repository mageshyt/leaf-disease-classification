import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from io import BytesIO
import base64
from fastapi import FastAPI, File, UploadFile
from PIL import Image

class ImagePredictor:
    def __init__(self, model, class_names, img_size=224):
        self.model = model
        self.class_names = class_names
        self.img_size = img_size

    def read_image(self, image_encode):

       pil_image = Image.open(image_encode)
       print(">> pill ",pil_image)
       return pil_image
    

    def process_image(self, image:Image.Image,img_shape=224):
        """
        Takes an image file path and turns the image into a tensor
        """
        # read the image file
        # image_file = tf.io.read_file(image_path)
        # decode the image
        img=tf.image.decode_image(image)

        # resize the image

        img_resize=tf.image.resize(img,size=[img_shape,img_shape])

        # rescale the image
        img_rescale=img_resize/255.

        return img_rescale

    def get_prediction_label(self, pred_proba):
        """
        Turns an array of prediction probabilities into a label
        """
        return self.class_names[pred_proba.argmax()]

    def predict(self, image:Image.Image):
        """
        Predicts the class label for the given image and displays the result.
        """
        # Get the prediction probabilities for the image

        img = self.process_image(image)
        pred_prob = self.model.predict(tf.expand_dims(img, axis=0))[0]

        # Get the predicted label
        pred_label = self.get_prediction_label(pred_prob)

        return {"prediction": pred_label, "max-probability": float(max(pred_prob)), "probabilities": pred_prob.tolist()}

    def display_prediction(self, image: Image.Image, pred_label, pred_prob):
        """
        Display the image along with the predicted label and probabilities.
        """
        image = self.process_image(image)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot the image
        ax1.imshow(image)
        ax1.axis('off')
        ax1.set_title(f'Prediction: {pred_label}')

        # Display the top 5 prediction probabilities as a bar chart
        top5_indices = np.argsort(pred_prob)[::-1][:5]
        top5_labels = [self.class_names[i] for i in top5_indices]
        top5_probs = pred_prob[top5_indices]
        ax2.barh(np.arange(5), top5_probs, color='purple')
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(5))
        ax2.set_yticklabels(top5_labels, size='small')
        ax2.set_xlim(0, 1.1)
        ax2.set_xlabel('Prediction Probability')
        ax2.set_title('Top 5 Predictions')

        plt.tight_layout()
        # plt.show()

        # Save the figure
        plot_bytes = BytesIO()
        fig.savefig(plot_bytes, format='png')
        plot_bytes.seek(0)

        # Close the figure to free up resources
        plt.close(fig)

        return plot_bytes

    
# Path: backend/server.py