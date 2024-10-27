import os
import time
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from torch.utils.data import DataLoader, Dataset

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

# Function to read file content based on file extension
def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    print(f"File extension: {ext}")  # Debugging line
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.readlines()
    else:
        raise ValueError("Unsupported file format")

# Start the timer
start_time = time.time()

# Prompt for file path
file_path = input("Enter the file path: ").strip('"')

# Read patient data from file
patient_data = read_file(file_path)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

# Check if GPU is available and move model to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create DataLoader for batch processing
batch_size = 105  # Adjust batch size as needed
dataset = TextDataset(patient_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Tokenize and get embeddings
embeddings = []
for batch in dataloader:
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model(**inputs)
    embeddings.extend(outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy())

# Summarize patient data using a summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
summarized_data = [summarizer(text, max_length=50, min_length=25, do_sample=False)[0]['summary_text'] for text in patient_data]

# Define categories and keywords
categories = {
    "Condition & Symptoms": ["diabetes", "hypertension", "symptoms", "pain", "edema"],
    "Medication & Dosage": ["medication", "tylenol", "aleve", "dosage"],
    "Procedures & Tests": ["procedure", "biopsy", "tests", "MCV", "RDW", "CBC"],
    "Body Part Involvement": ["heart", "liver", "kidney"],
    "History & Demographics": ["family history", "personal history", "demographics", "smoking"],
    "Duration": ["timeline", "months", "days", "admission"]
}

# Classify and summarize patient data based on categories
classified_data = {category: [] for category in categories}

for summary in summarized_data:
    for category, keywords in categories.items():
        if any(keyword in summary.lower() for keyword in keywords):
            classified_data[category].append(summary)

# Write categorized and summarized patient data to a text file in the same directory as the script
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(script_dir, f"SUMMARY_{os.path.basename(file_path)}.txt")
with open(output_file_path, "w", encoding='utf-8') as file:
    for category, summaries in classified_data.items():
        file.write(f"{category}:\n")
        for summary in summaries:
            file.write(f"• {summary}\n")
        file.write("\n")

# Confirmation message for GPU usage
if torch.cuda.is_available():
    print("Processing was done using the GPU.")
else:
    print("Processing was done using the CPU.")

# End the timer and print the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Categorized patient data has been written to {output_file_path}")
print(f"Time taken for processing: {elapsed_time:.2f} seconds")
