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

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


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

class LLaVAModel(ImageLLM):
    def __init__(self, 
        llava_model, 
        device=torch.device('cpu'), #TODO, migrate to accelerate to support multi gpu, multi batch training. Currently, only single gpu, single batch.
    ):
        super().__init__()
        self.device = device
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=llava_model,
            model_base=None,
            model_name=get_model_name_from_path(llava_model),
            device=device
        )        
        self.model.to(torch.float32)
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        self.normalize = torchvision.transforms.Normalize(
            mean = mean, std = std
        )
        
        self.inv_normalize = torchvision.transforms.Normalize(
            mean = [-m/s for m, s in zip(mean, std)],
            std = [1/s for s in std]
        )
        
        # for param in self.model.parameters():
        #     param.requires_grad = False
        
    def forward(self, tokens, image_tensor):
        L = tokens.shape[1]
        output = self.model(
            input_ids=tokens,
            images=self.normalize(image_tensor),
            image_sizes=self.image_sizes
        )
        return output.logits[:, -L:, :]
    
    def generate(self, raw_prompt, path_to_image, max_gen_len=64, temperature=0.1):
        image_files = [path_to_image]
        images = [Image.open(image_file).convert("RGB") for image_file in image_files]
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        images_tensor = self.normalize(self.inv_normalize(images_tensor))

        formatted_prompt = self.format_prompt(raw_prompt)
        formatted_prompt = self.add_prefix_to_formatted_propmt(formatted_prompt)
        input_ids = tokenizer_image_token(formatted_prompt, self.tokenizer, IMAGE_TOKEN_INDEX)
        input_ids = torch.tensor([input_ids], device=self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=None,
                num_beams=1,
                max_new_tokens=max_gen_len,
                use_cache=True,
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return outputs
    
    def generate_during_train(self, formatted_prompt, image_tensor, max_gen_len=64):
        formatted_prompt = self.add_prefix_to_formatted_propmt(formatted_prompt)
        input_ids = tokenizer_image_token(formatted_prompt, self.tokenizer, IMAGE_TOKEN_INDEX)
        input_ids = torch.tensor([input_ids], device=self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=self.normalize(image_tensor),
                image_sizes=self.image_sizes,
                do_sample=False,
                temperature=0,
                top_p=None,
                num_beams=1,
                max_new_tokens=max_gen_len,
                use_cache=True,
            )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    def preprocess_image_for_training(self, path_to_image):
        image_files = [path_to_image]
        images = [Image.open(image_file).convert("RGB") for image_file in image_files]
        self.image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device)
        images_tensor = self.inv_normalize(images_tensor)#.to(dtype=torch.float16)
        return images_tensor
    
    def save_trained_image_tensor(self, image_tensor, path_to_save_image):
        # image_tensor = torch.flip(image_tensor, [0])
        save_image(image_tensor, path_to_save_image)

    def add_prefix_to_formatted_propmt(self, formatted_prompt):
        conv_mode = "llava_v1"

        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], formatted_prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt
        
    def tokenize_prompt_target(self, formatted_prompt, target):
        formatted_prompt = self.add_prefix_to_formatted_propmt(formatted_prompt)
        
        prompt_tokens = tokenizer_image_token(formatted_prompt, self.tokenizer, IMAGE_TOKEN_INDEX)
        prompt_tokens_len = len(prompt_tokens)
        prompt_tokens_and_target = tokenizer_image_token(formatted_prompt + " " + target, self.tokenizer, IMAGE_TOKEN_INDEX)
        assert prompt_tokens_and_target[-1] != 2
        prompt_tokens_and_target.append(2)

        prompt_tokens_and_target = torch.tensor([prompt_tokens_and_target], device=self.device)
        target_tokens = prompt_tokens_and_target.clone()
        target_tokens[:, :prompt_tokens_len] = -100
        return prompt_tokens_and_target, target_tokens
    
    def twoway_tokenize_prompt_target(self, formatted_prompt, target1, target2):
        prompt_tokens = tokenizer_image_token(formatted_prompt, self.tokenizer, IMAGE_TOKEN_INDEX)
        prompt_tokens_len = len(prompt_tokens)
        prompt_tokens_and_target1 = tokenizer_image_token(formatted_prompt + " " + target1, self.tokenizer, IMAGE_TOKEN_INDEX)
        prompt_tokens_and_target1_len = len(prompt_tokens_and_target1)
        prompt_tokens_and_target1_and_target2 = tokenizer_image_token(formatted_prompt + " " + target1 + target2, self.tokenizer, IMAGE_TOKEN_INDEX)
        assert prompt_tokens_and_target1_and_target2[-1] != 2
        prompt_tokens_and_target1_and_target2.append(2)
        
        prompt_tokens_and_target1_and_target2 = torch.tensor([prompt_tokens_and_target1_and_target2], device=self.device)
        target_tokens = prompt_tokens_and_target1_and_target2.clone()
        target_tokens[:, :prompt_tokens_len] = -100
        
        return prompt_tokens_and_target1_and_target2, target_tokens, prompt_tokens_len, prompt_tokens_and_target1_len
    
    def format_prompt(self, raw_prompt):
        if type(raw_prompt) == str:
            return "<image>\n" + raw_prompt
        assert type(raw_prompt) == dict
        if 'output' in raw_prompt:
            raw_prompt = {k: v for k, v in raw_prompt.items() if k != 'output'}
        return "<image>\n" + raw_prompt["instruction"] + "\n" + raw_prompt["input"]
    

if __name__ == "__main__":
    model = LLaVAModel("liuhaotian/llava-v1.5-7b", device=torch.device('cuda:0'))
    prompt = "Describe the figure"
    image_file = "data/images_pp/llama_adapter/llama_adapter_logo.png"

    print(model.generate(prompt, image_file))