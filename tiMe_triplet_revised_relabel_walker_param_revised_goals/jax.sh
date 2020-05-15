# install jaxlib
PYTHON_VERSION=cp37  # alternatives: cp36, cp37, cp38
CUDA_VERSION=cuda101  # alternatives: cuda92, cuda100, cuda101, cuda102
PLATFORM=linux_x86_64  # alternatives: linux_x86_64
BASE_URL='https://storage.googleapis.com/jax-releases'
pip install --upgrade $BASE_URL/$CUDA_VERSION/jaxlib-0.1.43-$PYTHON_VERSION-none-$PLATFORM.whl

pip install --upgrade jax  # install jax
