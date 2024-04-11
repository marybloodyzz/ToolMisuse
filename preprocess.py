import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import torch
import random
from LLaMAAdapterModel import LLaMAAdapterModel
from tqdm import tqdm
import json
import math

from collections import defaultdict

device = "cpu"

model = LLaMAAdapterModel(path_to_llama_dir=PATH_TO_LLAMA_DIR, device=device)

for id in range(10):
    image = f"image{id}.png"
    image_tensor = model.preprocess_image_for_training(f"data/images/{image}").squeeze(0)
    model.save_trained_image_tensor(image_tensor, f"data/images_pp/llama_adapter/{image}")