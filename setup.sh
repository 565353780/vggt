cd ..
git clone https://github.com/565353780/camera-control.git
git clone https://github.com/jytime/LightGlue.git

pip install tqdm hydra-core omegaconf opencv-python \
  scipy onnxruntime requests matplotlib pillow \
  huggingface_hub einops safetensors ninja plyfile evo

pip install numpy==1.26.4
pip install gradio==5.17.1
pip install viser==0.2.23
pip install pydantic==2.10.6
pip install pycolmap==3.10.0
pip install pyceres==2.3

cd camera-control
./setup.sh

cd ../LightGlue
pip install .
