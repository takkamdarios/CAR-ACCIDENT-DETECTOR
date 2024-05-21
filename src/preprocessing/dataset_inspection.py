import os
from PIL import Image
import matplotlib.pyplot as plt

# Path to the dataset directory
dataset_path = 'dataset/images'

def inspect_images(dataset_path):
    """
    Inspect and display statistics about images in the dataset directory.
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')  # Add or remove file extensions as needed.
    image_paths = [os.path.join(dataset_path, fname)
                   for fname in os.listdir(dataset_path)
                   if fname.endswith(image_extensions)]

    # Initialize variables to collect information
    image_sizes = []
    aspect_ratios = []

    # Process each image file
    for image_path in image_paths:
        with Image.open(image_path) as img:
            # Collect image size and aspect ratio data
            width, height = img.size
            aspect_ratio = width / height
            image_sizes.append((width, height))
            aspect_ratios.append(aspect_ratio)

            # For now, we'll just print out the image path, size, and aspect ratio
            print(f"Image: {os.path.basename(image_path)}, Size: {img.size}, Aspect Ratio: {aspect_ratio:.2f}")

    # Plot the image sizes distribution if needed
    plt.scatter(*zip(*image_sizes))
    plt.title('Image Sizes Distribution')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.show()

    # Plot the aspect ratios distribution if needed
    plt.hist(aspect_ratios, bins=20)
    plt.title('Aspect Ratios Distribution')
    plt.xlabel('Aspect Ratio (Width/Height)')
    plt.ylabel('Frequency')
    plt.show()

# Call the function with the path to your dataset
inspect_images(dataset_path)
