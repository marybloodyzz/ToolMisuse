import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import torch
import random
from tqdm import tqdm
import json
import math

import data.attacks.Attack as Attack
from LLaMAAdapterModel import LLaMAAdapterModel

from collections import defaultdict
import transformers


attacks = {
    "delete_email_attack": Attack.delete_email_attack,
    "send_email1_attack": Attack.send_email1_attack,
    "send_email2_attack": Attack.send_email2_attack,
    "book_ticket_attack": Attack.book_ticket_attack,
    "query_website_attack": Attack.query_website_attack,
}

images = {
    "llama_adapter_logo": "data/images_pp/llama_adapter/llama_adapter_logo.png",
    "stable_diffusion": "data/images_pp/llama_adapter/stable_diffusion.png",
    "shutterstock_211440232": "data/images_pp/llama_adapter/shutterstock_211440232.png",
}


image_related_prompts = [x for x in open("data/prompts/image_related_questions.txt").read().split("\n")]
image_unrelated_prompts = [json.loads(x) for x in open("data/prompts/image_unrelated_questions.json").read().strip().split("\n")]
image_unrelated_prompts = [x for x in image_unrelated_prompts if not x["input"]]


image_related_prompts_test = [x for x in open("data/prompts/image_related_questions_test.txt").read().split("\n")]
image_unrelated_prompts_test = [json.loads(x) for x in open("data/prompts/image_unrelated_questions_test.json").read().strip().split("\n")]
image_unrelated_prompts_test = [x for x in image_unrelated_prompts_test if not x["input"]]

def get_one():
    if random.random() <= 0.15:
        return random.choice(image_related_prompts)
    else:
        return random.choice(image_unrelated_prompts)
    
def get_user_instruction(prompt_args):
    if type(prompt_args) == str:
        return prompt_args.strip()
    return prompt_args["instruction"].strip()

def log_image(model, image_tensor, step, path_to_log_dir):
    image_tensor = image_tensor.data.clone().detach().squeeze(0).cpu()
    print("img mean", image_tensor.mean(), "img max", image_tensor.max(), "img min", image_tensor.min())
    model.save_trained_image_tensor(image_tensor, os.path.join(path_to_log_dir, f"{step}.png"))
    return os.path.join(path_to_log_dir, f"{step}.png")
    
def l2_reg(x, y):
    x, y = x.view(1, 3, -1), y.view(1, 3, -1)
    #c1 = 8
    norm = torch.norm(x - y, p = 2, dim = 2)
    out = norm
    #print(out)
    return torch.sum(out)

def eval_attack(model, path_to_original_image, path_to_adversarial_image, attack_fn, small=True):
    model.eval()
    if small:
        related = image_related_prompts_test[: 10]
        unrelated = image_unrelated_prompts_test[: 10]
    else:
        related = image_related_prompts_test
        unrelated = image_unrelated_prompts_test
    metrics = {
        "related": defaultdict(list),
        "unrelated": defaultdict(list),
    }
    test_generations = {
        "related": {"input": [], "user_instruction": [], "output": []},
        "unrelated": {"input": [], "user_instruction": [], "output": []},
    }
    image_tensor = model.preprocess_image_for_training(path_to_original_image)
    for test_set, metric, generations in [(related, metrics["related"], test_generations["related"]), (unrelated, metrics["unrelated"], test_generations["unrelated"])]:
        
        for i, prompt_args in tqdm(enumerate(test_set), desc="evaluating"):
            prompt = model.format_prompt(prompt_args)
            user_instruction = get_user_instruction(prompt_args)
            
            model_output = model.generate(prompt_args, path_to_adversarial_image,  max_gen_len=256) # account for more tokens
            
            model_response, generated_attack = attack_fn.split_into_response_and_attack(model_output)
            prompt_tokens_and_target, target_tokens = model.tokenize_prompt_target(prompt, model_response.strip())
            loss_fn = torch.nn.CrossEntropyLoss()
            model.train()
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logits = model.forward(tokens=prompt_tokens_and_target, image_tensor=image_tensor)
                loss = loss_fn(logits.squeeze(0)[:-1], target_tokens.squeeze(0)[1:])
            if i == 0:
                print("user_instruction", user_instruction)
                print("model_output", model_output)
            generations["input"].append(prompt)
            generations["user_instruction"].append(user_instruction)
            generations["output"].append(model_output)
            eval_result = attack_fn.eval_attack(user_instruction, model_output)
            for k, v in eval_result.items():
                metric[k].append(v)
            metric['loss'].append(loss.item())
    metric_results = {
        "related": {
            
        },
        "unrelated": {
            
        }
    }
    print('')
    print('')
    print('=====RESULTS=====')
    for k1 in metrics.keys():
        for k2 in metrics[k1].keys():
            metric_results[k1][k2 + "_avg"] = sum(metrics[k1][k2]) / len(metrics[k1][k2])
            print(k1, k2 + "_avg", metric_results[k1][k2 + "_avg"])
    return metrics, metric_results, test_generations
            
mt, vt, max_vt = 0, 0, 0

def eval_mode(
    model, path_to_original_image, path_to_adversarial_image , attack_fn,
    save_path,
    sample, suffix
):
    metrics, metric_results, generations = eval_attack(model, path_to_original_image, path_to_adversarial_image, attack_fn, sample)
    with open(os.path.join(save_path, f"metrics_{suffix}.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    with open(os.path.join(save_path, f"metrics_results_{suffix}.json"), "w") as f:
        json.dump(metric_results, f, indent=4)
    with open(os.path.join(save_path, f"generations_{suffix}.json"), "w") as f:
        json.dump(generations, f, indent=4)

def attack(
    model, 
    path_to_image, 
    attack_fn,
    save_path,
    use_pgd=False,
    pgd_norm=2.0, 
    pgd_norm_threshold=4.,
    pgd_inf_clamp=0.05,
    l2_reg_lambda=0.05,
    step_size=0.01,
    max_steps=25000,
    accumulation=1, # first, lets simulate batch by gradient accumulation
    loss_lambda=1.0,
    load_image_tensor=None,
    use_adam=False,
    skip_training=False,
):
    path_to_original_image = path_to_image
    image_tensor = model.preprocess_image_for_training(path_to_image)
    log_image(model, image_tensor, "initial", save_path)
    image_tensor_fixed = image_tensor.clone().detach().requires_grad_(False)
    if load_image_tensor:
        image_tensor = model.preprocess_image_for_training(load_image_tensor)
        path_to_original_image = load_image_tensor
    model.train()
    
    loss_fn = torch.nn.CrossEntropyLoss()
    

    def adam(gradient, img, step, lr = step_size, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 0, amsgrad = False):
        gradient = gradient + weight_decay * img
        global mt
        global vt
        global max_vt
        mt = betas[0] * mt + (1 - betas[0]) * gradient
        vt = betas[1] * vt + (1 - betas[1]) * gradient**2

        bias_correction1 = 1 - betas[0] ** step
        bias_correction2 = 1 - betas[1] ** step
        step_size = lr / bias_correction1
        
        bias_correction2_sqrt = math.sqrt(bias_correction2)

        if amsgrad:
            if isinstance(max_vt, int):
                max_vt = vt
            else:
                max_vt = torch.maximum(max_vt, vt)
            denom = (max_vt.sqrt() / bias_correction2_sqrt) + eps
        else:
            denom = (vt.sqrt() / bias_correction2_sqrt) + eps
        return mt / denom * step_size

    pbar = tqdm(range(max_steps))
    last_eval_loss = None
    if not skip_training:
        model.zero_grad()
        for r in pbar:
            image_tensor_to_update = image_tensor.clone().detach().requires_grad_(True)
            prompt_args = get_one()
            prompt = model.format_prompt(prompt_args)
            user_instruction = get_user_instruction(prompt_args)
            model.eval()
            normal_answer = model.generate_during_train(prompt, image_tensor_fixed, max_gen_len=128)
            model.train()
            current_target = attack_fn.get_integrated_attack(user_instruction, normal_answer)
            if r == 0:
                print("prompt_args", prompt_args)
                print("prompt", prompt)
                print("user_instruction", user_instruction)
                print("normal_answer", normal_answer)
                print("current_target", current_target)
            
            prompt_tokens_and_target, target_tokens, prompt_len, first_part_len = model.twoway_tokenize_prompt_target(prompt, normal_answer, attack_fn.get_attack_string(user_instruction))
            with torch.cuda.amp.autocast():
                logits = model.forward(tokens=prompt_tokens_and_target, image_tensor=image_tensor_to_update)
            l = target_tokens.shape[1]
            l1 = loss_fn(logits.squeeze(0)[:-1][: first_part_len - 1], target_tokens.squeeze(0)[1:][: first_part_len - 1])
            l2 = loss_fn(logits.squeeze(0)[:-1][first_part_len - 1 :], target_tokens.squeeze(0)[1:][first_part_len - 1 :])
            loss = l1 * loss_lambda * (first_part_len - prompt_len) + l2 * (l - first_part_len)
            loss = loss / (l - prompt_len)
            if l2_reg_lambda != 0:
                loss = loss + l2_reg_lambda * l2_reg(image_tensor_to_update, image_tensor_fixed)
            loss = loss / accumulation
            loss.backward()
            if (r + 1) % accumulation == 0:
                with torch.no_grad():
                    if use_adam:
                        gradients = adam(image_tensor_to_update.grad, image_tensor, (r + 1) // accumulation)
                    else:
                        gradients = image_tensor_to_update.grad
                    gradients = image_tensor_to_update.grad * step_size
                    image_tensor -= gradients

                    delta = image_tensor.clone().detach() - image_tensor_fixed
                    
                    if use_pgd:
                        norm = torch.norm(delta.view(1, 3, -1), p = pgd_norm, dim = -1) # 3 color channels
                        for i in range(3):
                            if norm[0, i] > pgd_norm_threshold:
                                delta[:, i] = delta[:, i] * pgd_norm_threshold / norm[0, i]
                    if pgd_inf_clamp:
                        delta = delta.clamp(min=-pgd_inf_clamp, max=pgd_inf_clamp)

                    # make sure still within valid image range
                    image_tensor = image_tensor_fixed + delta    
                    image_tensor = image_tensor.clamp(min=0, max=1)
                model.zero_grad()
            
                torch.cuda.empty_cache()
            
            if (r + 1) % 1000 == 0:
                model.eval()
                path_to_adversarial_image = log_image(model, image_tensor, f"step_{r + 1}", save_path)
                eval_mode(
                    model, path_to_original_image, path_to_adversarial_image, attack_fn,
                    save_path,
                    True, f"step_{r + 1}"
                )
                model.train()
            pbar.set_description(f"train_loss: {loss.item()}; eval_loss: {last_eval_loss}")
    model.eval()
    path_to_adversarial_image = log_image(model, image_tensor, "final", save_path)
    eval_mode(
        model, path_to_original_image, path_to_adversarial_image, attack_fn,
        save_path,
        False, "final"
    )
    return image_tensor


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_model", type=str, required=True)
    parser.add_argument("--attack_fn", type=str, default="query_website_attack")
    parser.add_argument("--image_name", type=str, default="llama_adapter_logo")
    parser.add_argument("--save_path", type=str, default="trained/tmp")
    parser.add_argument("--use_pgd", action="store_true")
    parser.add_argument("--pgd_norm", type=float, default=2.0)
    parser.add_argument("--pgd_norm_threshold", type=float, default=4.0)
    parser.add_argument("--pgd_inf_clamp", type=float, default=0.00)
    parser.add_argument("--l2_reg_lambda", type=float, default=0.05)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=25000)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--loss_lambda", type=float, default=1.0)
    parser.add_argument("--use_adam", action="store_true")  
    parser.add_argument("--seed", default=42, type=int)
        
    args = parser.parse_args()
    transformers.set_seed(args.seed)
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = LLaMAAdapterModel(args.path_to_model, device=device)

    attack(
        model=model,
        path_to_image=images[args.image_name],
        attack_fn=attacks[args.attack_fn],
        save_path=args.save_path,
        use_pgd=args.use_pgd,
        pgd_norm=args.pgd_norm,
        pgd_norm_threshold=args.pgd_norm_threshold,
        pgd_inf_clamp=args.pgd_inf_clamp,
        l2_reg_lambda=args.l2_reg_lambda,
        step_size=args.step_size,
        accumulation=args.accumulation,
        loss_lambda=args.loss_lambda,
        max_steps=args.max_steps,
        use_adam=args.use_adam,
    )