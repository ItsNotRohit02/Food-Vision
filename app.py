import gradio as gr
import os
import torch

from model import create_ViT
from timeit import default_timer as timer
from typing import Tuple, Dict

with open("class_names.txt", "r") as f:
    class_names = [food_name.strip() for food_name in f.readlines()]

ViT_model, ViT_transforms = create_ViT(
    num_classes=126,
)

ViT_model.load_state_dict(
    torch.load(
        f="ViT.pth",
        map_location=torch.device("cpu"),
    )
)


def predict(img) -> Tuple[Dict, float]:
    start_time = timer()

    img = ViT_transforms(img).unsqueeze(0)

    ViT_model.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(ViT_model(img), dim=1)

    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

    pred_time = round(timer() - start_time, 5)

    return pred_labels_and_probs, pred_time


title = "FoodVision"
description = "An Vision Transformer feature extractor computer vision model to classify images of food into 126 different classes."
article = "Created by [Rohit](https://github.com/ItsNotRohit02)."

example_list = [["examples/" + example] for example in os.listdir("examples")]

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

demo.launch()
