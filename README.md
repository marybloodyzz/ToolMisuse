# VLMToolMisuse
This is the official code implementation for the paper `Misusing Tools in Large Language Models With Visual Adversarial Examples`.

## Environment
You can create a conda enviroment to run the code following `create_env.sh`.

## Description

Currently, we only support one Vision Language Model, the [LLaMAAdapter Model](https://github.com/OpenGVLab/LLaMA-Adapter). You need the original checkpoints of the LLaMA model downloaded, as specified [here](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal7b#setup).

Modeling: `model.py` and `LLaMAAdapterModel.py` contains code to support tuning the LLaMAAdapter Model.

Training: `train_adversarial_image.py` contains the code to conduct the adversarial training and evaluation. `run.sh` is an example execution script with some pre-defined hyper-parameters. 

Tool Use Templates: All our tool use templates are placed in `data/attacks/Attack.py`

Images: The original images are placed in `data/images/`. `data/images_pp/` contains the preprocessed images that will actually be used. Note that preprocessing could be different for different Vision Language Models. 

## Citation
If you find our code helpful, please consider citing

```
@article{fu2023misusing,
  title={Misusing Tools in Large Language Models With Visual Adversarial Examples},
  author={Fu, Xiaohan and Wang, Zihan and Li, Shuheng and Gupta, Rajesh K and Mireshghallah, Niloofar and Berg-Kirkpatrick, Taylor and Fernandes, Earlence},
  journal={arXiv preprint arXiv:2310.03185},
  year={2023}
}
```