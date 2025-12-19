cd ..
git clone https://github.com/jytime/LightGlue.git

pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

pip install tqdm hydra-core omegaconf opencv-python \
  scipy onnxruntime requests trimesh matplotlib \
  pillow huggingface_hub einops safetensors ninja

pip install gradio==5.17.1
pip install viser==0.2.23
pip install pydantic==2.10.6
pip install pycolmap==3.10.0
pip install pyceres==2.3

cd LightGlue
pip install .
