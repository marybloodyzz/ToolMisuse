conda create -n tool_misuse python=3.10
conda activate tool_misuse
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python tqdm bs4 sentencepiece openai-clip timm transformers
