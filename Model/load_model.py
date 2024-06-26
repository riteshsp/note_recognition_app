import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("second_model.h5")
# model = tf.keras.models.load_model("fourth.h5")


dic = {
    "0" : "10",
    "1" : "100",
    "2" : "20",
    "3" : "200",
    "4" : "2000",
    "5" : "50",
    "6" : "500",
    "7" : "Rupee Note not found"
}


def preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

target_size = (64, 64)  

# Preprocess the image
img_path = "archive/Indian currency dataset v1/test/50__147.jpg"
preprocessed_image = preprocess_image(img_path, target_size)


# Predict the class of the image
predictions = model.predict(preprocessed_image)


# Get the predicted class
predicted_class = np.argmax(predictions, axis=1)


# Print the predicted class
print("Predicted class:", predictions)
print("aaaaaaaaaaaaaaaaaaaa", predictions)


print("Predicted class:", dic[str(predicted_class[0])]) 

