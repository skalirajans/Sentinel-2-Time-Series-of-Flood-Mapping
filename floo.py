import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Load and preprocess data
post_flood_path = "C:\\project\\post flood"
pre_flood_path = "C:\\project\\pre flood"

def load_images(folder, check_post_flood=False):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            if check_post_flood and '_1' not in filename:
                continue
            img = Image.open(os.path.join(folder, filename))
            images.append(img)
    return images

pre_flood_images = load_images(pre_flood_path)
post_flood_images = load_images(post_flood_path, check_post_flood=True)

def convert_to_array(images, target_size=(128, 128)):
    image_arrays = []
    for img in images:
        img = img.resize(target_size)
        img_array = np.array(img)
        image_arrays.append(img_array)
    return np.array(image_arrays)

pre_flood_array = convert_to_array(pre_flood_images)
post_flood_array = convert_to_array(post_flood_images)

def generate_random_labels(size, n_classes=4):
    return np.random.randint(0, n_classes, size)

Y_labels = np.array([generate_random_labels((128, 128)) for _ in post_flood_array])
Y_labels = np.expand_dims(Y_labels, axis=-1)
Y_labels = to_categorical(Y_labels, num_classes=4)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

X_train, X_val, Y_train, Y_val = train_test_split(post_flood_array, Y_labels, test_size=0.2, random_state=42)

# U-Net Model
def unet(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.3)(c1)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.3)(c2)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.4)(c3)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.5)(c4)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.5)(c5)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    c5 = BatchNormalization()(c5)

    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.5)(c6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    c6 = BatchNormalization()(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.4)(c7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    c7 = BatchNormalization()(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.3)(c8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    c8 = BatchNormalization()(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.3)(c9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    c9 = BatchNormalization()(c9)

    outputs = Conv2D(4, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model
# Compile and train
model = unet(input_size=(128, 128, 3))
model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=5),
    validation_data=(X_val, Y_val),
    epochs=50,
    callbacks=callbacks
)
# Water Level Calculation
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

for i in range(len(post_flood_array)):
    pre_flood_image = pre_flood_array[i]
    post_flood_image = post_flood_array[i]
    predicted_mask = predictions[i] 
    water_flow = calculate_water_flow(pre_flood_image, predicted_mask)
    plt.figure(figsize=(15, 8))
    
    # Pre-Flood Image
    plt.subplot(1, 6, 1)
    plt.imshow(pre_flood_image)
    plt.title("Pre-Flood")
    plt.axis('off')

    # Post-Flood Image
    plt.subplot(1, 6, 2)
    plt.imshow(post_flood_image)
    plt.title("Post-Flood")
    plt.axis('off')

    # Water Bodies with Bounding Boxes
    plt.subplot(1, 6, 3)
    plt.imshow(highlight_class(predicted_mask, 1, color=(0, 0, 255)))  # Blue for water
    plt.title("Water\nBodies")
    
    plt.axis('off')

    # Human-Made Structures with Bounding Boxes
    plt.subplot(1, 6, 4)
    plt.imshow(highlight_class(predicted_mask, 2, color=(255, 0, 0)))  # Red for human-made structures
    plt.title("Man-Made\n Structures")
    plt.axis('off')

    # Trees with Bounding Boxes
    plt.subplot(1, 6, 5)
    plt.imshow(highlight_class(predicted_mask, 3, color=(0, 255, 0)))  # Green for trees
    plt.title("Trees")
    plt.axis('off')
    plt.subplot(1, 6, 6)  # Correct index for the last subplot
    plt.imshow(highlight_class(predicted_mask, 4, (0, 128, 0)))
    plt.title("Detected Forest Areas")
    plt.axis('off')  
    plt.show()
    plt.close()