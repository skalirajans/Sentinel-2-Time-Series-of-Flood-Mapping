Sentinel-2-Time-Series-of-Flood-Mapping

Date: 03-02-2025

🎯 AIM:

To process multitemporal Sentinel-2 satellite images captured before and after flooding, perform image segmentation to detect water-level changes, extract optical flow direction, and visualize the progression of flood water using a hybrid CNN and ConvLSTM U-Net architecture.

🧠 Algorithm
1. 📥 Loading Satellite Images

Input:

File locations of pre-flood and post-flood satellite images in formats: .png, .jpg, .jpeg, .tif

Process:

List image filenames in folders

Filter valid image formats

Load images using PIL and convert them to RGB for consistency

def load_images(folder, check_post_flood=False):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tif')):
            if check_post_flood and '_1' not in filename:
                continue
            img = Image.open(os.path.join(folder, filename)).convert("RGB")
            images.append(img)
    return images


Output:
List of image objects ready for analysis or processing.

2. 🔁 Converting Images into Numpy Arrays

Input:

Loaded PIL images

Process:

Resize each image to (128 x 128) pixels

Convert resized image into NumPy array for model compatibility

def convert_to_array(images, target_size=(128, 128)):
    image_arrays = [np.array(img.resize(target_size)) for img in images]
    return np.array(image_arrays)


Output:
NumPy array with the shape (n_samples, 128, 128, 3).

3. 🏷️ Generating Dynamic Labels Using Optical Flow

Input:

Pre-flood and post-flood NumPy images

Optical flow algorithm

Process:

Compute pixel displacement between consecutive time frames using Farneback Optical Flow.

From the flow magnitude at each pixel, generate class labels.

Use thresholding to detect high water movement regions.

Convert output to one-hot encoding (4 classes).

Y_labels = np.array([generate_labels(cv2.calcOpticalFlowFarneback(
    cv2.cvtColor(pre_flood_array[i].astype(np.uint8), cv2.COLOR_RGB2GRAY),
    cv2.cvtColor(post_flood_array[i].astype(np.uint8), cv2.COLOR_RGB2GRAY),
    None, 0.5, 3, 15, 3, 5, 1.2, 0)) for i in range(len(post_flood_array))])


Output:
Categorical segmentation masks indicating levels of flood spread.

4. 📊 Preparing Dataset with Train-Test Split & Data Augmentation

Input:

Post-flood images (X)

Generated flood masks (Y)

Process:

Split data into 80% training and 20% validation

Use augmentation (shifts, rotations, flips) to improve generalization

X_train, X_val, Y_train, Y_val = train_test_split(post_flood_array, Y_labels, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)


Output:
Train and validation batches for model training.

5. 🧱 Hybrid CNN-ConvLSTM UNet Model Creation

Input:

Tensor input layer accepting (128, 128, 3) images

Process:

Encoder (CNN): Extract spatial features

Bottleneck using ConvLSTM2D: Preserve temporal dependencies

Decoder: Upsample segmented output back to image size

def hybrid_rnn_cnn(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    
    # Encoder, Bottleneck, Decoder code here...
    
    outputs = Conv2D(4, (1, 1), activation='softmax')(x)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


Output:
Fully compiled segmentation model using Adam optimizer and categorical_crossentropy loss.

6. ⚙️ Model Training

Process:

Fit model with augmented training data

Implement early stopping and LR reduction on plateau

history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=8),
    validation_data=(X_val, Y_val),
    epochs=20,
    callbacks=callbacks
)


Output:
Training and validation logs. Model weights captured.

7. 📈 Training Visualization

Process:

Plot accuracy and loss over epochs

plt.plot(history.history['accuracy'], ...)
plt.plot(history.history['loss'], ...)


Output:
Visual curves to understand model performance.

8. 🌊 Optical Flow & Flood Direction Visualization

Input:

Pre-flood and post-flood image pairs

Process:

Compute flow using Farneback method

Plot flow vectors over post-flood image

def visualize_flow(post_img, flow):
    ...
    ax.quiver(...)


Output:
Quiver plot showing water movement vector fields.

9. 💧 Water Level Change Detection

Process:

Compare water segmentation mask before and after flood

Measure pixel-level growth

def calculate_water_level_change(pre_mask, post_mask):
    ...
    return np.sum(post_water) - np.sum(pre_water)


Output:
Pixel count of water expansion for each image.

10. 📦 Bounding Box Visualization of Flooded Regions

Process:

Extract contours of flood segments

Draw bounding boxes

def highlight_class(predicted_mask, class_id, color=(0, 0, 255)):
    ...


Output:
Visual highlight of regions affected by flood.

 Description of Python and TensorFlow Implementation
Image Loading & Preprocessing

Uses PIL and NumPy for consistent formatting

Works across multi-format satellite inputs

🔍 Optical Flow Computation

cv2.calcOpticalFlowFarneback to measure water movement

🧠 Model Architecture

Combination of CNN & ConvLSTM supports spatial+temporal processing

Inspired by U-Net for segmentation tasks

🎨 Visualization

matplotlib for image, quiver vectors

cv2 for contour extraction and flood boundaries
