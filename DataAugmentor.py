import cv2
import os
import random
import shutil

# Define the path to the original folder and the new folder
original_folder = "Train"
new_folder = "NewTrain"

# Define the maximum number of images per subfolder
max_images_per_subfolder = 900

# Define a list of augmentation functions
def augment_image(image):
    angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, 0, 0, 0]
    angle = angles[random.randint(0, 4)]
    rotated_image = image

    if angle != 0:
        rotated_image = cv2.rotate(image, angle)

    for i in range(0,random.randint(1,2)):
        mirrored_image = cv2.flip(image, 1)

    # Randomly resize the image to a scale between 0.8 and 1.2 times its original size
    scale = random.uniform(0.8, 1.2)
    resized_image = cv2.resize(mirrored_image, None, fx=scale, fy=scale)

    # Randomly adjust brightness and contrast
    alpha = random.uniform(0.7, 1.3)  # Brightness factor
    beta = random.randint(-30, 30)  # Contrast adjustment
    augmented_image = cv2.convertScaleAbs(resized_image, alpha=alpha, beta=beta)

    return augmented_image

# Create the new folder if it doesn't exist
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# Iterate over subfolders in the original folder
for root, dirs, files in os.walk(original_folder):
    for dir in dirs:
        original_subfolder = os.path.join(root, dir)
        new_subfolder = os.path.join(new_folder, os.path.relpath(original_subfolder, original_folder))

        if not os.path.exists(new_subfolder):
            os.makedirs(new_subfolder)

        # Collect a list of image file paths in the original subfolder
        image_paths = [os.path.join(original_subfolder, filename) for filename in os.listdir(original_subfolder) if filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

        # Calculate the number of images needed for repetitions
        num_images_needed = max_images_per_subfolder

        # Ensure that each image is used at least once before repetition
        num_images_needed = max(num_images_needed, len(image_paths))

        for i in range(num_images_needed):
            # Get the image path from the list (loop back to the beginning if necessary)
            image_path = image_paths[i % len(image_paths)]

            # Load the image
            image = cv2.imread(image_path)

            if image is not None:
                # Apply augmentation
                augmented_image = augment_image(image)

                # Define the new image path
                new_image_path = os.path.join(new_subfolder, f"{i}_{os.path.basename(image_path)}")

                # Save the augmented image to the new folder
                cv2.imwrite(new_image_path, augmented_image)

# Print a message when the augmentation is complete
print("Augmentation complete.")
