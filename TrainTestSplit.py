import cv2
import os
import random
import shutil

# Define the path to the original folder, train folder, and test folder
original_folder = "images"
train_folder = "Train"
test_folder = "Test"

# Define the percentage of images to allocate to the test set
test_percentage = 0.1

# Create the train and test folders if they don't exist
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# Iterate over subfolders in the original folder
for root, dirs, files in os.walk(original_folder):
    for dir in dirs:
        original_subfolder = os.path.join(root, dir)
        train_subfolder = os.path.join(train_folder, os.path.relpath(original_subfolder, original_folder))
        test_subfolder = os.path.join(test_folder, os.path.relpath(original_subfolder, original_folder))

        if not os.path.exists(train_subfolder):
            os.makedirs(train_subfolder)
        if not os.path.exists(test_subfolder):
            os.makedirs(test_subfolder)

        # Collect a list of image file paths in the original subfolder
        image_paths = [os.path.join(original_subfolder, filename) for filename in os.listdir(original_subfolder) if filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Shuffle the image list
        random.shuffle(image_paths)

        # Calculate the number of images to allocate to the test set
        num_test_images = int(len(image_paths) * test_percentage)

        # Split the images into train and test sets
        test_images = image_paths[:num_test_images]
        train_images = image_paths[num_test_images:]

        # Copy test images to the test subfolder
        for image_path in test_images:
            test_image_name = os.path.basename(image_path)
            test_image_destination = os.path.join(test_subfolder, test_image_name)
            shutil.copy(image_path, test_image_destination)

        # Copy train images to the train subfolder
        for image_path in train_images:
            train_image_name = os.path.basename(image_path)
            train_image_destination = os.path.join(train_subfolder, train_image_name)
            shutil.copy(image_path, train_image_destination)

# Print a message when the splitting is complete
print("Splitting into train and test sets complete.")
