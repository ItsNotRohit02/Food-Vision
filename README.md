# Food-Vision
FoodVision can be found [here](https://huggingface.co/spaces/ItsNotRohit/FoodVision).

## Overview
* **Model Architecture:** Google Vision Transformer (ViT) - base patch16-224
* **Custom Dataset:** 121 Foods with 1000 images each, can be found [here](https://huggingface.co/datasets/ItsNotRohit/Food121).
* **Training Time:** 5 hours and 20 minutes on a Tesla T4 GPU in Google Colab.
* **Accuracy:** Achieved an impressive accuracy of 90.2% on the Test split of the custom dataset.

## HyperParameters
* **Optimizer:** Adam with betas = (0.9, 0.999) and epsilon = 1e-08 
* **Loss Function:** CrossEntropyLoss with label smoothing = 0.1
* **Scheduler:** Linear
* **Epochs:** 5
