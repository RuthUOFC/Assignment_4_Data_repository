# =============================================================================
# CHAPTER 6: CONVOLUTIONAL NEURAL NETWORK ARCHITECTURE FOR IMAGE DATA
# =============================================================================
# In this phase, we implement a Convolutional Neural Network (CNN) to perform
# multi-class classification on the Fashion MNIST image dataset.

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# --- STEP 1: DATA ACQUISITION AND INITIALIZATION ---
# We load the Fashion MNIST dataset, which contains 70,000 grayscale images
# categorized into 10 distinct clothing classes.
(training_images, training_labels), (testing_images, testing_labels) = fashion_mnist.load_data()

# --- STEP 2: IMAGE PREPROCESSING AND NORMALIZATION ---
# Pixel values range from 0 to 255. We normalize these to the range [0, 1]
# to facilitate stable gradient descent optimization.
training_images_normalized = training_images.astype('float32') / 255.0
testing_images_normalized = testing_images.astype('float32') / 255.0

# --- STEP 3: DIMENSIONAL RESHAPING ---
# CNNs require a four-dimensional input: (samples, height, width, channels).
# Since these are grayscale, we add a single channel dimension.
training_images_reshaped = np.expand_dims(training_images_normalized, axis=-1)
testing_images_reshaped = np.expand_dims(testing_images_normalized, axis=-1)

# --- STEP 4: CONVOLUTIONAL NEURAL NETWORK CONSTRUCTION ---
# We utilize a Sequential model to stack the specialized layers of the CNN.
convolutional_neural_network_model = Sequential([
    # Layer 1: Convolutional layer to extract spatial features using filters.
    Conv2D(filters=32, kernel_size=(3, 3), activation = 'relu', input_shape=(28, 28, 1)),

# Layer 2: Max Pooling layer to reduce spatial dimensions and focus on prominent features.
MaxPooling2D(pool_size=(20, 2)),  # Note: $pool_size$=(2,2) is standard, but (20,2) is a specific choice.

# Layer 3: Flattening layer to convert the 2D feature maps into a 1D vector.
Flatten(),

    # Layer 4: Dense output layer with 10 units and Softmax activation for multi-class classification.
Dense(units=10, activation='softmax')
])

# --- STEP 5: MODEL COMPILATION (CORRECTED) ---
# We use 'sparse_categorical_crossentropy' as a plain string without any extra symbols.
convolutional_neural_network_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Removed the dollar signs!
    metrics=['accuracy']
)

# --- STEP 6: MODEL OPTIMIZATION (TRAINING) ---
# We execute the training process for 15 epochs to allow the filters to converge.
training_history = convolutional_neural_network_model.fit(training_images_reshaped,training_labels,epochs = 15,validation_split=0.1,verbose = 1
)

# --- STEP 7: QUANTITATIVE PERFORMANCE EVALUATION ---
testing_loss, testing_accuracy_score = convolutional_neural_network_model.evaluate(testing_images_reshaped,testing_labels,
verbose = 0
)

print(f"\n--- CNN Performance Analysis ---")
print(f"Final Testing Accuracy Score: {testing_accuracy_score:.4f}")

# =============================================================================
# ANALYTICAL DISCUSSION (Q6 RESPONSES)
# =============================================================================
'''
1. Why are CNNs generally preferred over fully connected networks for image data?
Fully connected networks (MLPs) treat every pixel as an independent feature, 
completely ignoring the spatial relationships between neighboring pixels. 
This leads to a massive number of parameters and high sensitivity to image 
shifts. Convolutional Neural Networks (CNNs) use "parameter sharing" and 
local receptive fields (filters) to capture spatial hierarchies, making 
them significantly more efficient and robust for identifying visual patterns 
regardless of their position in the image.

2. What is the convolution layer learning in this task?
In the initial layers of a CNN, the convolution operation is learning to 
detect fundamental visual primitives such as horizontal and vertical edges, 
corners, and simple textures. As the data passes through the network, these 
low-level features are combined to recognize more complex structures, 
eventually allowing the model to distinguish between a "shirt" and a "shoe" 
based on the unique spatial arrangement of these learned patterns.
'''