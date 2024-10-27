# Rutgers-Health-Hack-2024
# Version of python used
https://www.python.org/downloads/release/python-3120/

# Create your project directory
in terminal
mkdir my_project(Or whatever you'd like to name it) <br />
cd my_project(Or whatever you'd you named it)

# Install Torch
pip install torch <br />

# (Optional) Install CUDA Development Kit - version 12.4 recommended <br />
https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local 

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

# Testing
First run the bert.py file so that it can download any need information from (https://huggingface.co/medicalai/ClinicalBERT) without running into any major issues. <br />
Then run clinical_bert_test.py as a demo test to make sure that the AI model is functioning properly. <br />

# Running
To run our summarizing script, run 
