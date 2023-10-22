import gradio as gr
import os
import torch

from model import create_ViT
from timeit import default_timer as timer
from typing import Tuple, Dict

# Setup class names
with open("class_names.txt", "r") as f:
    class_names = [food_name.strip() for food_name in  f.readlines()]


# Create model
ViT_model, ViT_transforms = create_ViT(
    num_classes=126,
)

# Load saved weights
ViT_model.load_state_dict(
    torch.load(
        f="ViT.pth",
        map_location=torch.device("cpu"),
    )
)


# Create predict function
def predict(img) -> Tuple[Dict, float]:

    start_time = timer()

    # Transform the target image and add a batch dimension
    img = ViT_transforms(img).unsqueeze(0)

    # Put model into evaluation mode and turn on inference mode
    ViT_model.eval()
    with torch.inference_mode():
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(ViT_model(img), dim=1)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)

    # Return the prediction dictionary and prediction time
    return pred_labels_and_probs, pred_time


##GRADIO APP
# Create title, description and article strings
title = "FoodVision"
description = "An Vision Transformer feature extractor computer vision model to classify images of food into 126 different classes."
article = "Created by [Rohit](https://github.com/ItsNotRohit02)."

# Create examples list from "examples/" directory
example_list = [["examples/" + example] for example in os.listdir("examples")]

# Create Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predictions"),
        gr.Number(label="Prediction time (s)"),
    ],
    examples=example_list,
    title=title,
    description=description,
    article=article,
)

# Launch the app!
demo.launch()
