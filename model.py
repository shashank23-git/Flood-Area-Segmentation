from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = load_model('flood_segmentation_unet.h5', compile=False)
# Read image
img = cv2.imread('test.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize to model input size
img_resized = cv2.resize(img, (256, 256))

# Normalize
img_input = img_resized / 255.0
img_input = np.expand_dims(img_input, axis=0)

# Predict
pred = model.predict(img_input)

# Convert prediction to binary mask
mask = (pred[0, :, :, 0] > 0.5).astype(np.uint8)

# Display
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(img_resized)
plt.title("Original Image")

plt.subplot(1,2,2)
plt.imshow(mask, cmap='gray')
plt.title("Predicted Flood Mask")

plt.show()
