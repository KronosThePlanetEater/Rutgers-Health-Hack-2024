# Rutgers Health Hack 2024

# Version of python used
https://www.python.org/downloads/release/python-3120/

# Create your project directory
In terminal run the following <br />
![image](https://github.com/user-attachments/assets/0688d129-7bb7-4712-bc7e-8fa33a65934d)

mkdir my_project(Or whatever you'd like to name it) <br />
cd my_project(Or whatever you'd you named it) <br />
myenv\Scripts\activate  # On Windows

# Install Torch
pip install torch <br />

# Install CUDA Development Kit - version 12.4 recommended (for NVIDIA GPUs) <br />
https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local 

# Install cuDNN (for NVIDIA GPUs) <br />
py -m pip install nvidia-cudnn-cu12

# Install PyTorch with CUDA support (for NVIDIA GPUs) <br />
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install Hugging Face Transformers
pip install transformers

# Install additional libraries 
pip install python-docx docx2txt PyPDF2 hl7 <br />
pip install sentencepiece (Run only once). <br />
pip install sacremoses (run only once). <br />

# Verify CUDA and cuDNN Installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('cuDNN version:', torch.backends.cudnn.version()); print('CUDA device count:', torch.cuda.device_count()); print('CUDA device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"

# Testing
First run the bert.py file so that it can download any need information from (https://huggingface.co/medicalai/ClinicalBERT) without running into any major issues. <br />
Then run clinical_bert_test.py as a demo test to make sure that the AI model is functioning properly. <br />

# Running
To run our summarizing script, run BridgeScript.py, in the terminal it will prompt you with the file path for the data/notes you want to summarize. Copy the file as path, and paste it into the terminal (Note: remove the quotes at the ends of the copied path or else it will throw an error).

To run the other our summarizing script, run BridgeScriptHighlighting.py, in the terminal it will prompt you with the file path for the data/notes you want to summarize. Copy the file as path, and paste it into the terminal (Note: remove the quotes at the ends of the copied path or else it will throw an error). BridgeScriptHighlighting.py is meant to summarize the information and create links to the part of the data/note it got the summarized information from. This is to make it easier for people to quickly read the summarized information and then open up the more detialed part of the patient information that they are intrested in. 

#Translate 
pip install sentencepiece (Run only once). <br />
pip install sacremoses (run only once). <br />
Similar proccess as to running the other two scripts, when prompted, input the path of the file you want to be translated. It will translate that file in export it to the project folder.

# Output
Once the script is done running, in the terminal, it will output some text "Categorized patient data has been written to C:\Users\dhrum\Health-hack\SUMMARY_H and P 1 Commentary.txt.txt" (may vary based on folder installation) This is where the summarized file is located. <br />
It also outputs the amount of time it took the run the script, which can be used to optimize it further or better prformance. 

# Optimization
While the either of BridgeScript.py or BridgeScriptHighlighting.py are running, if on Windows run task-manager and under the performance tab you will see how much of the GPU memory is being used. 
![image](https://github.com/user-attachments/assets/78edbbb3-efe4-4e2a-81f3-c4cfee952404) <br />
You can use this information to change the "batch_size" number as need so it can run well on your hardware. Higher the batch number the more resources it will use, but run faster. Lower the batch number the less resources it will use, but run slower. (The amount of Vram utilized will vary based on the number)
![image](https://github.com/user-attachments/assets/4ff74b68-10e8-4e8f-8bb3-3c8c68bf4212)


# Tools used
https://www.python.org/downloads/release/python-3120/ <br />
Hugging face transformer <br />
torch  <br />
https://huggingface.co/docs/transformers/model_doc/marian <br />
https://huggingface.co/medicalai/ClinicalBERT <br />
https://pytorch.org/get-started/locally/ <br />
https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local  (CUDA Development Kit) <br />
https://developer.nvidia.com/cudnn (nvidia-cudnn-cu12) <br />
[Visual Studio Code](https://code.visualstudio.com/)
