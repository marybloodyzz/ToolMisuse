python train_adversarial_image.py \
    --path_to_model <PATH_TO_YOUR_DOWNLOADED_LLAMA_2_7B_MODEL> \
    --attack_fn send_email2_attack \
    --l2_reg_lambda 0.05 \
    --loss_lambda 0.1 \
    --step_size 0.01 \
    --accumulation 1 \
    --max_steps 12000 \
    --use_adam \
    --save_path tmp
