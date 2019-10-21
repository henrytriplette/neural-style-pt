# Python 3.6
pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.1-cp36-cp36m-win_amd64.whl
pip3 install torchvision

# Verify
python3
import torch
torch.cuda.is_available()

# Display cards
nvidia-smi

# Run
python3 neural_style.py -style_image <image.jpg> -content_image <image.jpg>

python neural_style.py -style_image style.jpg -content_image input.jpg -output_image output.png -gpu 0 -backend cudnn -cudnn_autotune -optimizer adam -num_iterations 500

python neural_style.py -style_image style.jpg -content_image input.jpg -output_image output.png -model_file models/nin_imagenet.pth -gpu 0 -backend cudnn -cudnn_autotune -optimizer adam -num_iterations 500