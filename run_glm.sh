# export TRANSFORMERS_CACHE=/data/lizhe/.cache \
# export XDG_CACHE_HOME=/data/lizhe/.cache \

# load clip download_root = "/data/lizhe/.cache"

python train_adversarial_image.py \
    --path_to_model "THUDM/glm-4v-9b" \
    --attack_fn send_email2_attack \
    --l2_reg_lambda 0.05 \
    --loss_lambda 0.1 \
    --step_size 0.01 \
    --accumulation 1 \
    --max_steps 100 \
    --use_adam \
    --save_path tmp
