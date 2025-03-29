
# X-ray Image Classification Using Keras

This project demonstrates how to train a deep learning model for X-ray image classification using Keras. The task is to classify images into two categories: **NORMAL** and **PNEUMONIA**. The model is trained on a dataset of chest X-ray images. https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Project Overview

- **Data:** Chest X-ray images divided into three categories: **train**, **validation**, and **test**.
- **Model:** Convolutional Neural Network (CNN) implemented using Keras.
- **Libraries Used:** Keras, TensorFlow, Python, NumPy, Matplotlib.

## Prerequisites

Before starting, ensure that the following libraries are installed:

```bash
pip install tensorflow
pip install matplotlib
pip install scikit-learn
pip install streamlit
```

## Dataset Structure

The dataset should be organized as follows:

```
data/
    chest_xray/
        train/
            NORMAL/
            PNEUMONIA/
        val/
            NORMAL/
            PNEUMONIA/
        test/
            NORMAL/
            PNEUMONIA/
```

### 1. **Verify Dataset**

First, you need to verify that the dataset is structured correctly. You can use the following script to check the number of images in each category:

```python
import os

def verify_dataset(path):
    for folder in ['train', 'val', 'test']:
        normal_dir = os.path.join(path, folder, 'NORMAL')
        pneumonia_dir = os.path.join(path, folder, 'PNEUMONIA')
        
        normal_images = len(os.listdir(normal_dir))
        pneumonia_images = len(os.listdir(pneumonia_dir))
        
        print(f"{folder.capitalize()} Data:")
        print(f"NORMAL in {folder}: {normal_images} images")
        print(f"PNEUMONIA in {folder}: {pneumonia_images} images")
        print()

verify_dataset('data/chest_xray')
```

### 2. **Model Building**

We use a Convolutional Neural Network (CNN) in Keras to build our model. Here's the architecture of the model:

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Create the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

### 3. **Data Preprocessing**

We will preprocess the images by resizing them to 150x150 pixels and using `ImageDataGenerator` for augmenting the images:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess data
train_data = train_datagen.flow_from_directory(
    'data/chest_xray/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

val_data = val_datagen.flow_from_directory(
    'data/chest_xray/val',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)
```

### 4. **Model Training**

Now, train the model using the preprocessed data:

```python
history = model.fit(
    train_data,
    steps_per_epoch=train_data.samples // train_data.batch_size,
    epochs=10,
    validation_data=val_data,
    validation_steps=val_data.samples // val_data.batch_size
)
```

### 5. **Evaluate the Model**

After training the model, you can evaluate its performance on the test data:

```python
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    'data/chest_xray/test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_loss, test_acc = model.evaluate(test_data)
print(f"Test accuracy: {test_acc}")
```

### 6. **Model Saving**

You can save the trained model for later use:

```python
model.save('xray_model.h5')
```

### 7. **Deploy the Model with Streamlit**

To deploy the model and create a simple web application, use Streamlit:

```python
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = load_model('xray_model.h5')

st.title("X-ray Image Classification")
uploaded_file = st.file_uploader("Choose an X-ray image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict the class
    prediction = model.predict(img_array)
    
    if prediction < 0.5:
        st.write("Prediction: NORMAL")
    else:
        st.write("Prediction: PNEUMONIA")
```

### 8. **Run Streamlit App**

To run the Streamlit app:

```bash
streamlit run src/WebApp.py
```

### 9. **Conclusion**

This setup provides an easy way to build and deploy an image classification model using Keras and TensorFlow for medical image analysis. You can extend this application further by integrating more complex models, improving the UI, or deploying it on a web server.

---

**License:**  
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Acknowledgements:**  
The dataset used in this project is from [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).
```

### Key Points:
1. **Dataset Structure:** Includes how to organize the dataset into training, validation, and test sets.
2. **Model Building:** Code for building a simple CNN model using Keras.
3. **Data Preprocessing:** Steps for preprocessing images using `ImageDataGenerator`.
4. **Training:** Shows how to train the model with the preprocessed data.
5. **Model Evaluation:** Includes the evaluation step on the test dataset.
6. **Model Saving:** Instructions on saving the trained model for future use.
7. **Deployment with Streamlit:** Provides a simple web interface for classifying uploaded X-ray images.

