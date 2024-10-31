# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from imgaug import augmenters as iaa
import os

# Function to load an image
def load_image(path):
    return Image.open(path)

# Function to display images
def display_images(original, augmented, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(augmented)
    axes[1].set_title(title)
    axes[1].axis("off")
    plt.show()

# Load an example image (replace 'your_image_path.jpg' with your image file path)
image_path = 'your_image_path.jpg'
image = load_image(image_path)
image_np = np.array(image)

# Define data augmentations using imgaug
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5),                 # 50% chance of horizontal flip
    iaa.Flipud(0.2),                 # 20% chance of vertical flip
    iaa.Affine(rotate=(-25, 25)),    # Rotate image between -25 and 25 degrees
    iaa.Multiply((0.8, 1.2)),        # Change brightness by 20%
    iaa.GaussianBlur(sigma=(0, 1.0)) # Apply Gaussian blur with sigma up to 1.0
])

# Apply augmentation
augmented_image_np = augmentation(image=image_np)
augmented_image = Image.fromarray(augmented_image_np)

# Display the original and augmented images
display_images(image, augmented_image, "Augmented Image")

# Further Augmentations and Saving Examples (Optional)
augmented_images_folder = 'augmented_images'
os.makedirs(augmented_images_folder, exist_ok=True)

for i in range(5):  # Create 5 augmented versions of the image
    augmented_img_np = augmentation(image=image_np)
    augmented_img = Image.fromarray(augmented_img_np)
    augmented_img.save(os.path.join(augmented_images_folder, f"augmented_image_{i+1}.jpg"))
