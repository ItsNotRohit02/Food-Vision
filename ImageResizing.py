import cv2
import os

# Define the path to the original folder and the new folder
original_folder = "Train"
new_folder = "NewTrain"

# Define the maximum dimension
max_dimension = 512

# Create the new folder if it doesn't exist
if not os.path.exists(new_folder):
    os.makedirs(new_folder)


# Function to resize an image while maintaining aspect ratio
def resize_image(image, max_dimension):
    height, width = image.shape[:2]

    if max(height, width) <= max_dimension:
        return image  # No need to resize if smaller than max dimension

    if height > width:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    else:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


# Iterate over subfolders in the original folder
for root, dirs, files in os.walk(original_folder):
    for dir in dirs:
        original_subfolder = os.path.join(root, dir)
        new_subfolder = os.path.join(new_folder, os.path.relpath(original_subfolder, original_folder))

        if not os.path.exists(new_subfolder):
            os.makedirs(new_subfolder)

        # Collect a list of image file paths in the original subfolder
        image_paths = [os.path.join(original_subfolder, filename) for filename in os.listdir(original_subfolder) if
                       filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for image_path in image_paths:
            # Load the image
            image = cv2.imread(image_path)

            if image is not None:
                # Resize the image while maintaining aspect ratio
                resized_image = resize_image(image, max_dimension)

                # Define the new image path
                new_image_path = os.path.join(new_subfolder, os.path.basename(image_path))

                # Save the resized image to the new folder
                cv2.imwrite(new_image_path, resized_image)

# Print a message when the resizing is complete
print("Resizing complete.")
