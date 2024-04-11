import torch
import sys
import types

# load image
import cv2
from PIL import Image 

def load_image(image_path):
    return Image.fromarray(cv2.imread(image_path))

# save image
from torchvision.utils import save_image


class ImageLLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, tokens, image_tensor):
        '''
        Given a sequence of tokens and the required image tensor, return the output logits of passing the models
        '''
        raise NotImplementedError
    
    def generate(self, raw_prompt, path_to_image, temperature=0.1):
        '''
        Obtain the generated output given the propmt and the path to the image. This matches the real use scenario
        '''
        raise NotImplementedError
    
    def generate_during_train(self, formatted_prompt, image_tensor, max_gen_len):
        '''
        Obtain the generated output given the prompt and the image tensor. This is used during training
        '''
        raise NotImplementedError
    
    def preprocess_image_for_training(self, path_to_image, max_gen_len):
        '''
        Given the path to the image, return the image tensor that can be used for training.
        No normalization is expected, such that a real image should correspond to a tensor of valaus [0, 1].
        '''
        raise NotImplementedError
    
    def save_trained_image_tensor(self, image_tensor, path_to_save_image):
        '''
        Save the image tensor to a path, expected to undo (rotations/flipping) in the preprocessing.
        '''
        raise NotImplementedError
    
    def tokenize_prompt_target(self, formatted_prompt, target):
        '''
        Return the tokens of the prompt + target
        Returns both the full prompt + target tokens, and the target tokens with the prompt tokens masked
        '''
        raise NotImplementedError

    def twoway_tokenize_prompt_target(self, formatted_prompt, target1, target2):
        '''
        Return the tokens of the prompt + target1 + target2
        Returns both the full prompt + target1 + target2 tokens, and the target tokens with the prompt tokens masked,
        and the length of target1 tokens and the length of target2 tokens.
        '''
        raise NotImplementedError
    
    def format_prompt(self, raw_prompt):
        '''
        Formats the raw prompt into the format that the model expects
        '''
        raise NotImplementedError
