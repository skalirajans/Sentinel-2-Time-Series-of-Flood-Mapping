import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Reshape, BatchNormalization, Dropout, Input, UpSampling2D, ConvLSTM2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load images
post_flood_path = "C:\\project\\post flood"
pre_flood_path = "C:\\project\\pre flood"

def load_images(folder, check_post_flood=False):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            if check_post_flood and '_1' not in filename:
                continue
            img = Image.open(os.path.join(folder, filename)).convert("RGB")
            images.append(img)
    return images

pre_flood_images = load_images(pre_flood_path)
post_flood_images = load_images(post_flood_path, check_post_flood=True)

def convert_to_array(images, target_size=(128, 128)):
    image_arrays = [np.array(img.resize(target_size)) for img in images]
    return np.array(image_arrays)

pre_flood_array = convert_to_array(pre_flood_images)
post_flood_array = convert_to_array(post_flood_images)

# Generate Labels (Assume water movement as flow direction labels)
def generate_labels(flow_data, num_classes=4):
    h, w = flow_data.shape[:2]
    labels = np.zeros((h, w, 1), dtype=np.uint8)
    magnitude = np.sqrt(flow_data[..., 0]**2 + flow_data[..., 1]**2)
    labels[magnitude > np.percentile(magnitude, 75)] = 1  # Mark high movement regions
    return to_categorical(labels, num_classes=num_classes)

Y_labels = np.array([generate_labels(cv2.calcOpticalFlowFarneback(
    cv2.cvtColor(pre_flood_array[i].astype(np.uint8), cv2.COLOR_RGB2GRAY),
    cv2.cvtColor(post_flood_array[i].astype(np.uint8), cv2.COLOR_RGB2GRAY),
    None, 0.5, 3, 15, 3, 5, 1.2, 0)) for i in range(len(post_flood_array))])

# Split dataset
X_train, X_val, Y_train, Y_val = train_test_split(post_flood_array, Y_labels, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Hybrid RNN-CNN Model
def hybrid_rnn_cnn(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    
    # Encoder
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Bottleneck
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    # ConvLSTM Layer
    x = Reshape((1, 32, 32, 256))(x)
    x = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=True)(x)
    x = ConvLSTM2D(filters=128, kernel_size=(3, 3), padding='same', return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    # Decoder
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    x = UpSampling2D((2, 2))(x) 
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    outputs = Conv2D(4, (1, 1), activation='softmax')(x)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Compile and Train model
model = hybrid_rnn_cnn()
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=8),
    validation_data=(X_val, Y_val),
    epochs=20,
    callbacks=callbacks
)
# Display results for each image in a single cha
# Plot Accuracy and Loss Graph
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='best')

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')

plt.tight_layout()
plt.show()

# Visualization of Flow
def visualize_flow(post_img, flow):
    h, w = flow.shape[:2]
    y, x = np.mgrid[0:h:10, 0:w:10].reshape(2, -1)
    fx, fy = flow[y, x].T
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(post_img)
    ax.quiver(x, y, fx, fy, color='blue', angles='xy', scale_units='xy', scale=1.5, width=0.0025)
    plt.title("Water Flow Direction (Post-Flood)")
    plt.axis('off')
    plt.show()

def calculate_optical_flow(pre_flood_image, post_flood_image):
    gray_pre = cv2.cvtColor(pre_flood_image, cv2.COLOR_RGB2GRAY)
    gray_post = cv2.cvtColor(post_flood_image, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray_pre, gray_post, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

# Predict and visualize
for i in range(len(post_flood_array)):
    pre_flood_image = pre_flood_array[i].astype(np.uint8)
    post_flood_image = post_flood_array[i].astype(np.uint8)
    flow = calculate_optical_flow(pre_flood_image, post_flood_image)
    visualize_flow(post_flood_image, flow)

def calculate_water_level_change(pre_mask, post_mask):
    pre_water = (np.argmax(pre_mask, axis=-1) == 1).astype(np.uint8)
    post_water = (np.argmax(post_mask, axis=-1) == 1).astype(np.uint8)
    water_change = np.sum(post_water) - np.sum(pre_water)
    return water_change

for i in range(len(post_flood_array)):
    pre_flood_image = pre_flood_array[i]
    post_flood_image = post_flood_array[i]
    predicted_mask = model.predict(np.expand_dims(post_flood_image, axis=0))[0]
    water_level_change = calculate_water_level_change(Y_labels[i], predicted_mask)
    print(f"Water Level Change for Image {i+1}: {water_level_change} pixels")

# Predict and visualize
predictions = model.predict(post_flood_array)

def calculate_water_flow(pre_image, post_image):
    pre_water_mask = (np.argmax(pre_image, axis=-1) == 1).astype(np.uint8)
    post_water_mask = (np.argmax(post_image, axis=-1) == 1).astype(np.uint8)
    
    # Compute flow difference
    flow_diff = cv2.subtract(post_water_mask, pre_water_mask)
    return flow_diff

# Visualization with bounding boxes
def highlight_class(predicted_mask, class_id, color=(0, 0, 255)):
    mask = (np.argmax(predicted_mask, axis=-1) == class_id).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    highlighted_image = cv2.cvtColor(post_flood_image, cv2.COLOR_RGB2BGR)
    cv2.drawContours(highlighted_image, contours, -1, color, 2)
    return cv2.cvtColor(highlighted_image, cv2.COLOR_BGR2RGB)

# Water flow tracking function
def calculate_water_flow(pre_image, post_image):
    pre_water_mask = (np.argmax(pre_image, axis=-1) == 1).astype(np.uint8)
    post_water_mask = (np.argmax(post_image, axis=-1) == 1).astype(np.uint8)
    
    # Compute flow difference
    flow_diff = cv2.subtract(post_water_mask, pre_water_mask)
    return flow_diff

for i in range(len(post_flood_array)):
    pre_flood_image = pre_flood_array[i]
    post_flood_image = post_flood_array[i]
    predicted_mask = predictions[i] 
    water_flow = calculate_water_flow(pre_flood_image, predicted_mask)
    
    plt.figure(figsize=(15, 8))
    plt.subplot(1, 6, 1)
    plt.imshow(pre_flood_image)
    plt.title("Pre-Flood")
    plt.axis('off')
    
    plt.subplot(1, 6, 2)
    plt.imshow(post_flood_image)
    plt.title("Post-Flood")
    plt.axis('off')
    
    plt.subplot(1, 6, 3)
    plt.imshow(highlight_class(predicted_mask, 1, color=(0, 0, 255)))
    plt.title("Water\nBodies")
    plt.axis('off')
    plt.show()
    plt.close()