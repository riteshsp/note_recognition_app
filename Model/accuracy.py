import tensorflow as tf
import numpy as np

from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras._tf_keras.keras.models import Sequential, load_model, save_model

# from tensorflow.keras.models import Sequential, load_model, save_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array



# Load the trained model
model = load_model('currency_recognition_model.h5')

# Define the data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    rotation_range=30,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    "/home/user/Documents/learning/AI python/Tensorflow/archive/Indian currency dataset v1/training",
    target_size=(64, 64),
    batch_size=32,
    shuffle=True,
    seed=42,
    color_mode="rgb",
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "/home/user/Documents/learning/AI python/Tensorflow/archive/Indian currency dataset v1/validation/",
    target_size=(64, 64),
    batch_size=32,
    shuffle=False,  # Important to keep shuffle=False for accurate evaluation
    seed=42,
    color_mode="rgb",
    class_mode='categorical')

# Decode class indices
class_indices = train_generator.class_indices
class_indices = {v: k for k, v in class_indices.items()}

# Evaluate the model on the validation dataset
validation_steps = test_generator.samples // test_generator.batch_size
eval_results = model.evaluate(test_generator, steps=validation_steps, verbose=1)

# Print overall accuracy
overall_accuracy = eval_results[1]
print(overall_accuracy)
print(f'Overall accuracy: {overall_accuracy * 100:.2f}%')

# Get the predictions for the validation dataset
predictions = model.predict(test_generator, steps=validation_steps)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Calculate accuracy per class
from collections import defaultdict
correct_predictions = defaultdict(int)
total_predictions = defaultdict(int)

for true, pred in zip(true_classes, predicted_classes):
    total_predictions[true] += 1
    if true == pred:
        correct_predictions[true] += 1

# Print accuracy per class
for class_index, class_name in class_indices.items():
    accuracy = (correct_predictions[class_index] / total_predictions[class_index]) * 100
    print(f'Accuracy for class {class_name}: {accuracy:.2f}%')