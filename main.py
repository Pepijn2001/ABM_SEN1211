import csv
from model import FloorPlanModel
import mesa
import pandas as pd
from PIL import Image
import numpy as np


def image_to_floor_plan(image_path, threshold=128):
    """
    Convert an image-based floor plan to a matrix representation.

    Args:
        image_path (str): Path to the floor plan image file.
        threshold (int): Pixel intensity threshold for binarization.

    Returns:
        np.ndarray: A 2D numpy array representing the floor plan.
                    0 = walkable, 1 = obstacle.
    """
    # Load the image
    img = Image.open(image_path)

    # Convert to grayscale
    grayscale_img = img.convert("L")  # "L" mode is 8-bit grayscale

    # Convert to numpy array
    img_array = np.array(grayscale_img)

    # Binarize the image based on the threshold
    floor_plan = (img_array < threshold).astype(int)  # 1 for obstacles, 0 for walkable

    return floor_plan

floor_plan_dir = "./assets/aula_plattegrond.png"
floor_plan = image_to_floor_plan(floor_plan_dir)

# Run the model and see the results
model = FloorPlanModel(floor_plan, 100, 5) # run the model with 30x30 grid and 100 agents and a vision of 5
for i in range(100): # run the model for 100 steps
    model.step() # step the model by 1

# Collect the data from the model
model_data = model.datacollector.get_model_vars_dataframe()
agent_data = model.datacollector.get_agent_vars_dataframe()