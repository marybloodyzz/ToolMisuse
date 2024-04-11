import torch
import sys
import types

from model import ImageLLM, load_image, save_image

# llama adapter
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


def llama_adapter_forward_inference_all_return(self, visual_query, tokens, start_pos: int):
    _bsz, seqlen = tokens.shape
    h = self.llama.tok_embeddings(tokens)
    freqs_cis = self.llama.freqs_cis.to(h.device)
    freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
    mask = None
    mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
    mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

    for layer in self.llama.layers[:-1 * self.query_layer]:
        h = layer(h, start_pos, freqs_cis, mask)

    adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
    adapter_index = 0

    for layer in self.llama.layers[-1 * self.query_layer:]:
        dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
        dynamic_adapter = dynamic_adapter + visual_query
        h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
        adapter_index = adapter_index + 1

    h = self.llama.norm(h)
    output = self.llama.output(h)

    return output

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class LLaMAAdapterModel(ImageLLM):
    def __init__(self, 
        path_to_llama_dir, 
        device=torch.device('cpu'), #TODO, migrate to accelerate to support multi gpu, multi batch training. Currently, only single gpu, single batch.
    ):
        super().__init__()
        self.device = device
        self.base_model, self.base_preprocess = llama_adapter.load("BIAS-7B", llama_dir=path_to_llama_dir, device=device)
        self.base_model.forward_inference_all_return = types.MethodType(llama_adapter_forward_inference_all_return, self.base_model)
        
        n_px = 224
        self.preprocess_no_normalize = Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
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
        
    def forward(self, tokens, image_tensor):
        with torch.cuda.amp.autocast():
            visual_query = self.base_model.forward_visual(self.normalize(image_tensor))
            logits = self.base_model.forward_inference_all_return(visual_query=visual_query, tokens=tokens, start_pos=0)
        return logits
    
    def generate(self, raw_prompt, path_to_image, max_gen_len=64, temperature=0.1):
        image = load_image(path_to_image)
        image_tensor = self.base_preprocess(image).unsqueeze(0).to(self.device)
        formatted_prompt = self.format_prompt(raw_prompt)
        with torch.no_grad():
            generated_text = self.base_model.generate(image_tensor, [formatted_prompt], max_gen_len=max_gen_len, temperature=temperature)[0]
        return generated_text
    
    def generate_during_train(self, formatted_prompt, image_tensor, max_gen_len=64):
        with torch.no_grad():
            generated_text = self.base_model.generate(self.normalize(image_tensor), [formatted_prompt], max_gen_len=max_gen_len)[0]
        return generated_text

    def preprocess_image_for_training(self, path_to_image):
        image = load_image(path_to_image)
        image_tensor = self.preprocess_no_normalize(image).unsqueeze(0).to(self.device)
        return image_tensor
    
    def save_trained_image_tensor(self, image_tensor, path_to_save_image):
        image_tensor = torch.flip(image_tensor, [0])
        save_image(image_tensor, path_to_save_image)
        
    def tokenize_prompt_target(self, formatted_prompt, target):
        prompt_tokens = self.base_model.tokenizer.encode(formatted_prompt, bos=True, eos=False)
        prompt_tokens_len = len(prompt_tokens)
        prompt_tokens_and_target = self.base_model.tokenizer.encode(formatted_prompt + " " + target, bos=True, eos=True)
        prompt_tokens_and_target = torch.tensor([prompt_tokens_and_target], device=self.device)
        target_tokens = prompt_tokens_and_target.clone()
        target_tokens[:, :prompt_tokens_len] = -100
        return prompt_tokens_and_target, target_tokens
    
    def twoway_tokenize_prompt_target(self, formatted_prompt, target1, target2):
        prompt_tokens = self.base_model.tokenizer.encode(formatted_prompt, bos=True, eos=False)
        prompt_tokens_len = len(prompt_tokens)
        prompt_tokens_and_target1 = self.base_model.tokenizer.encode(formatted_prompt + " " + target1, bos=True, eos=False)
        prompt_tokens_and_target1_len = len(prompt_tokens_and_target1)
        prompt_tokens_and_target1_and_target2 = self.base_model.tokenizer.encode(formatted_prompt + " " + target1 + target2, bos=True, eos=True)
        
        prompt_tokens_and_target1_and_target2 = torch.tensor([prompt_tokens_and_target1_and_target2], device=self.device)
        target_tokens = prompt_tokens_and_target1_and_target2.clone()
        target_tokens[:, :prompt_tokens_len] = -100
        
        return prompt_tokens_and_target1_and_target2, target_tokens, prompt_tokens_len, prompt_tokens_and_target1_len
    
    def format_prompt(self, raw_prompt):
        if type(raw_prompt) == str:
            return llama_adapter.format_prompt(raw_prompt)
        assert type(raw_prompt) == dict
        if 'output' in raw_prompt:
            raw_prompt = {k: v for k, v in raw_prompt.items() if k != 'output'}
        return llama_adapter.format_prompt(**{k: v for k, v in raw_prompt.items() if v})
    
    