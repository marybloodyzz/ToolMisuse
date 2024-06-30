
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from huggingface_hub import login
login(token="your token")

# import os
# os.environ["TRANSFORMERS_CACHE"] = "enter if needed"

device = "cuda:3" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4v-9b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
            "THUDM/glm-4v-9b",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)

image = Image.open("test2.jpg").convert('RGB')
image_tensor = tokenizer.apply_chat_template([{"role": "user", "image": image}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)["images"]
prompt = "hello"
prompt_tokens = tokenizer.encode(prompt)
prompt_tokens = torch.tensor([prompt_tokens], device=device)

image_tensor_to_update = image_tensor.clone().detach().requires_grad_(True)

with torch.cuda.amp.autocast():
    tokens = prompt_tokens.to(device)
    image_tensor = image_tensor.to(device)

    output = model.forward(
                input_ids=prompt_tokens,
                images=image_tensor,
                return_dict=True
            )

    logits = output.logits
