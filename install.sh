sudo apt update && sudo apt install python3-pip docker.io -y
pip install transformers torch einops accelerate xformers fastapi uvicorn --no-warn-script-location
python3 -c "from transformers.utils import default_cache_path as dcp; print(dcp)"
