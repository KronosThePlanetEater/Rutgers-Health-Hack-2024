# Rutgers-Health-Hack-2024

# Create your project directory
mkdir my_project <br />
cd my_project

# Install PyTorch
pip install torch <br />

# (Optional) Install CUDA Development Kit - version 12.4 recommended <br />
# Install cuDNN <br />
py -m pip install nvidia-cudnn-cu12

# Install PyTorch with CUDA support (for NVIDIA GPUs) <br />
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Hugging Face Transformers
pip install transformers

# Install additional libraries 
pip install python-docx docx2txt PyPDF2 hl7

# Verify CUDA and cuDNN Installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('cuDNN version:', torch.backends.cudnn.version()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
