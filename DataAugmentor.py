import cv2
import os
import random
import shutil

original_folder = "Train"
new_folder = "NewTrain"

max_images_per_subfolder = 900


def augment_image(image):
    angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, 0, 0, 0]
    angle = angles[random.randint(0, 4)]
    rotated_image = image
    if angle != 0:
        rotated_image = cv2.rotate(image, angle)

    for i in range(0, random.randint(1, 2)):
        mirrored_image = cv2.flip(rotated_image, 1)

    scale = random.uniform(0.8, 1.2)
    resized_image = cv2.resize(mirrored_image, None, fx=scale, fy=scale)

    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    augmented_image = cv2.convertScaleAbs(resized_image, alpha=alpha, beta=beta)

    return augmented_image


if not os.path.exists(new_folder):
    os.makedirs(new_folder)

for root, dirs, files in os.walk(original_folder):
    for dir in dirs:
        original_subfolder = os.path.join(root, dir)
        new_subfolder = os.path.join(new_folder, os.path.relpath(original_subfolder, original_folder))

        if not os.path.exists(new_subfolder):
            os.makedirs(new_subfolder)

        image_paths = [os.path.join(original_subfolder, filename) for filename in os.listdir(original_subfolder) if
                       filename.lower().endswith(('.jpg', '.jpeg', '.png'))]
        num_images_needed = max_images_per_subfolder
        num_images_needed = max(num_images_needed, len(image_paths))

        for i in range(num_images_needed):
            image_path = image_paths[i % len(image_paths)]
            image = cv2.imread(image_path)

            if image is not None:
                augmented_image = augment_image(image)
                new_image_path = os.path.join(new_subfolder, f"{i}_{os.path.basename(image_path)}")
                cv2.imwrite(new_image_path, augmented_image)

print("Augmentation complete.")
