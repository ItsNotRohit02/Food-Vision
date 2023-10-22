import cv2
import os
import random
import shutil

original_folder = "images"
train_folder = "Train"
test_folder = "Test"

test_percentage = 0.1

if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

for root, dirs, files in os.walk(original_folder):
    for dir in dirs:
        original_subfolder = os.path.join(root, dir)
        train_subfolder = os.path.join(train_folder, os.path.relpath(original_subfolder, original_folder))
        test_subfolder = os.path.join(test_folder, os.path.relpath(original_subfolder, original_folder))

        if not os.path.exists(train_subfolder):
            os.makedirs(train_subfolder)
        if not os.path.exists(test_subfolder):
            os.makedirs(test_subfolder)

        image_paths = [os.path.join(original_subfolder, filename) for filename in os.listdir(original_subfolder) if
                       filename.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(image_paths)
        num_test_images = int(len(image_paths) * test_percentage)

        test_images = image_paths[:num_test_images]
        train_images = image_paths[num_test_images:]

        for image_path in test_images:
            test_image_name = os.path.basename(image_path)
            test_image_destination = os.path.join(test_subfolder, test_image_name)
            shutil.copy(image_path, test_image_destination)

        for image_path in train_images:
            train_image_name = os.path.basename(image_path)
            train_image_destination = os.path.join(train_subfolder, train_image_name)
            shutil.copy(image_path, train_image_destination)

print("Splitting into train and test sets complete.")
