cat <<< '
Bootstrap: docker                              # Start from a Docker container
From: nvcr.io/nvidia/pytorch:20.08-py3                     # Specify the starting container from Docker Hub

%files                                         # Copy files from the host system into the container
  $(pwd)   /home/nerf-ngp

%post
  apt -y update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends  coreutils
  source ~/.bashrc
  cd tiny-cuda-nn/bindings/torch && python setup.py install
  pip3 install -r /home/nerf-ngp/torch-ngp/requirements.txt
  cd /home/nerf-ngp/torch-ngp && bash scripts/install_ext.sh
  echo "Done Setting Up Environment"

%help
	This container runs  object nerf
' > object_nerf.def