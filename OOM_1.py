
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from huggingface_hub import login
login(token="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")
from train_adversarial_image import *
attack_fn=attacks["send_email2_attack"]
# import os
# os.environ["TRANSFORMERS_CACHE"] = "enter if needed"
def format_prompt(raw_prompt):
    if type(raw_prompt) == str:
        return tokenizer.apply_chat_template([{"role": "user", "content": raw_prompt}],
                                    add_generation_prompt=True, tokenize=False,
                                    )
    assert type(raw_prompt) == dict
    if 'output' in raw_prompt:
        raw_prompt = {k: v for k, v in raw_prompt.items() if k != 'output'}
        raw_prompt_str = str(raw_prompt)
    return raw_prompt_str

def get_user_instruction(prompt_args):
    if type(prompt_args) == str:
        return prompt_args.strip()
    return prompt_args["instruction"].strip()

def generate_during_train(formatted_prompt, image_tensor, max_gen_len=64):
    image = Image.open("test2.jpg").convert('RGB')
    inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": formatted_prompt}],
                                    add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                    return_dict=True)
    
    inputs["images"] = image_tensor
    inputs = inputs.to(device)
    gen_kwargs = {"max_length": max_gen_len, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(outputs[0])

def twoway_tokenize_prompt_target(formatted_prompt, target1, target2):
        # Encode prompt with manual addition of BOS/EOS if they exist
        image = Image.open("test2.jpg").convert('RGB')
        prompt_tokens = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": formatted_prompt}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)["input_ids"]
        prompt_tokens_len = len(prompt_tokens)
        prompt_tokens_and_target1 = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": (formatted_prompt + " " + target1)}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)["input_ids"]
        prompt_tokens_and_target1_len = len(prompt_tokens_and_target1)
        
  
        prompt_tokens_and_target1_and_target2 = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": (formatted_prompt + " " + target1 + " " + target2)}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)["input_ids"]
        prompt_tokens_and_target1_and_target2 = prompt_tokens_and_target1_and_target2.to(device)        
        target_tokens = prompt_tokens_and_target1_and_target2.clone()
        target_tokens[:, :prompt_tokens_len] = -100
        print(target_tokens.shape)
        print(prompt_tokens_and_target1_and_target2.shape)
        return prompt_tokens_and_target1_and_target2, target_tokens, prompt_tokens_len, prompt_tokens_and_target1_len
       


device = "cuda:7" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4v-9b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
            "THUDM/glm-4v-9b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)

image = Image.open("data/images_pp/llama_adapter/llama_adapter_logo.png").convert('RGB')
image_tensor = tokenizer.apply_chat_template([{"role": "user", "image": image}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)["images"]
model.train()
prompt_args = {'instruction': 'Compute', 'input': '', 'output': 'The area of a rectangle can be calculated by multiplying its length by its width. In this case, the length of the rectangle is given as 10 cm and the width as 5 cm. Therefore, the area of the rectangle with the given dimensions is `10 cm x 5 cm = 50 cm²`.'}
prompt = format_prompt(prompt_args)
user_instruction = get_user_instruction(prompt_args)
model.eval()
normal_answer = generate_during_train(prompt, image_tensor, max_gen_len=128)
model.train()

prompt_tokens_and_target, _, _, _ = twoway_tokenize_prompt_target(prompt, normal_answer, attack_fn.get_attack_string(user_instruction))

image_tensor_to_update = image_tensor.clone().detach().requires_grad_(True)
for param in model.parameters():
    param.requires_grad = False

with torch.cuda.amp.autocast():
    tokens = prompt_tokens_and_target.to(device)
    image_tensor_to_update = image_tensor_to_update.to(device)
    count = 0
    output = model.forward(
                input_ids=tokens,
                images=image_tensor_to_update,
                return_dict=True
            )

    logits = output.logits
