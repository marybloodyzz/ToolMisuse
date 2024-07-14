import torch
import sys
import types
import clip
from model import ImageLLM, save_image, load_image
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from torchvision import transforms
from huggingface_hub import login
login(token="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")

import llama_adapter

# customize transform
from PIL import Image 
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC



def _convert_image_to_rgb(image):
    return image.convert("RGB")

class glmModel(ImageLLM):
    def __init__(self, 
        path_to_glm_dir, 
        device=torch.device('cpu'), #TODO, migrate to accelerate to support multi gpu, multi batch training. Currently, only single gpu, single batch.
    ):
        super().__init__()
        
        self.clip_transform = clip.load('ViT-L/14', download_root = "/data/lizhe/.cache") #glm added


        self.device = device
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "THUDM/glm-4v-9b",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4v-9b", trust_remote_code=True)
        # self.tokenizer.image_size = 224

        self.base_preprocess = self.clip_transform
        
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        self.normalize = torchvision.transforms.Normalize(
            mean = mean, std = std
        )
        
        self.inv_normalize = torchvision.transforms.Normalize(
            mean = [-m/s for m, s in zip(mean, std)],
            std = [1/s for s in std]
        )

        for param in self.base_model.parameters():
            param.requires_grad = False
        # print("printing np sizes: ")
        # for n, p in self.base_model.named_parameters():
        #     print(n, p.size(), p.requires_grad)

        
    def forward(self, tokens, image_tensor):
        for param in self.base_model.parameters():
            param.requires_grad = False
        L = tokens.shape[1]
        # with torch.no_grad():
        with torch.cuda.amp.autocast():
            tokens = tokens.to(self.device)
            image_tensor = image_tensor.to(self.device)
            output = self.base_model.forward(
                input_ids=tokens,
                images=image_tensor,
                return_dict=True
            )
        return output.logits[:, -L:, :]
    
    
    def generate_during_train(self, formatted_prompt, image_tensor, max_gen_len=64):
        image = Image.open("test2.jpg").convert('RGB')
        inputs = self.tokenizer.apply_chat_template([{"role": "user", "image": image, "content": formatted_prompt}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)
        
        inputs["images"] = image_tensor
        inputs = inputs.to(self.device)
        gen_kwargs = {"max_length": max_gen_len, "do_sample": True, "top_k": 1}
        with torch.no_grad():
            outputs = self.base_model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(outputs[0])
        
            

    def preprocess_image_for_training(self, path_to_image):
        print(path_to_image)
        image = Image.open(path_to_image).convert('RGB')
        image_tensor = self.tokenizer.apply_chat_template([{"role": "user", "image": image}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)["images"]

        return image_tensor


    def save_trained_image_tensor(self, image_tensor, path_to_save_image):
        image_tensor = torch.flip(image_tensor, [0])
        save_image(image_tensor, path_to_save_image)
        
    def tokenize_prompt_target(self, formatted_prompt, target):
        image = Image.open("test2.jpg").convert('RGB')
        prompt_tokens = self.tokenizer.apply_chat_template([{"role": "user", "image": image, "content": formatted_prompt}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)["input_ids"]
        prompt_tokens_len = len(prompt_tokens)
        prompt_tokens_and_target = self.tokenizer.apply_chat_template([{"role": "user", "image": image, "content": (formatted_prompt + " " + target)}],
                                add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                return_dict=True)["input_ids"]

        prompt_tokens_and_target = prompt_tokens_and_target.to(self.device)
        target_tokens = prompt_tokens_and_target.clone()
        target_tokens[:, :prompt_tokens_len] = -100
        return prompt_tokens_and_target, target_tokens
    
    def twoway_tokenize_prompt_target(self, formatted_prompt, target1, target2):
        # Encode prompt with manual addition of BOS/EOS if they exist
        image = Image.open("test2.jpg").convert('RGB')
        prompt_tokens = self.tokenizer.apply_chat_template([{"role": "user", "image": image, "content": formatted_prompt}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)["input_ids"]
        prompt_tokens_len = len(prompt_tokens)
        prompt_tokens_and_target1 = self.tokenizer.apply_chat_template([{"role": "user", "image": image, "content": (formatted_prompt + " " + target1)}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)["input_ids"]
        prompt_tokens_and_target1_len = len(prompt_tokens_and_target1)
        
  
        prompt_tokens_and_target1_and_target2 = self.tokenizer.apply_chat_template([{"role": "user", "image": image, "content": (formatted_prompt + " " + target1 + " " + target2)}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)["input_ids"]
        prompt_tokens_and_target1_and_target2 = prompt_tokens_and_target1_and_target2.to(self.device)        
        target_tokens = prompt_tokens_and_target1_and_target2.clone()
        target_tokens[:, :prompt_tokens_len] = -100
        print(target_tokens.shape)
        print(prompt_tokens_and_target1_and_target2.shape)
        return prompt_tokens_and_target1_and_target2, target_tokens, prompt_tokens_len, prompt_tokens_and_target1_len
       

    
    def format_prompt(self, raw_prompt):
        if type(raw_prompt) == str:
            return self.tokenizer.apply_chat_template([{"role": "user", "content": raw_prompt}],
                                       add_generation_prompt=True, tokenize=False,
                                       )
        assert type(raw_prompt) == dict
        if 'output' in raw_prompt:
            raw_prompt = {k: v for k, v in raw_prompt.items() if k != 'output'}
            raw_prompt_str = str(raw_prompt)
        return raw_prompt_str


    
