cat <<< '
source activate torch-ngp
export PATH=${PATH}:/usr/local/cuda-11.8/bin
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_PATH=/usr/local/cuda-11.8
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-11.8/lib64
' > ~/.bashrc
source ~/.bashrc
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r /home/nerf-ngp/torch-ngp/requirements.txt
cd tiny-cuda-nn/bindings/torch && python setup.py install
cd /home/nerf-ngp/torch-ngp && bash scripts/install_ext.sh