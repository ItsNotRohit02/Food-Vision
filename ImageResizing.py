import cv2
import os

original_folder = "Train"
new_folder = "NewTrain"

max_dimension = 512

if not os.path.exists(new_folder):
    os.makedirs(new_folder)


def resize_image(image, max_dimension):
    height, width = image.shape[:2]

    if max(height, width) <= max_dimension:
        return image

    if height > width:
        new_height = max_dimension
        new_width = int(width * (max_dimension / height))
    else:
        new_width = max_dimension
        new_height = int(height * (max_dimension / width))

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


for root, dirs, files in os.walk(original_folder):
    for dir in dirs:
        original_subfolder = os.path.join(root, dir)
        new_subfolder = os.path.join(new_folder, os.path.relpath(original_subfolder, original_folder))

        if not os.path.exists(new_subfolder):
            os.makedirs(new_subfolder)

        image_paths = [os.path.join(original_subfolder, filename) for filename in os.listdir(original_subfolder) if
                       filename.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for image_path in image_paths:
            image = cv2.imread(image_path)

            if image is not None:
                resized_image = resize_image(image, max_dimension)
                new_image_path = os.path.join(new_subfolder, os.path.basename(image_path))
                cv2.imwrite(new_image_path, resized_image)

print("Resizing complete.")
