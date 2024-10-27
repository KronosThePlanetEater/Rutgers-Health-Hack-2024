from transformers import AutoTokenizer, AutoModel, pipeline

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

# Example patient data
patient_data = [
    "Patient has a history of diabetes and hypertension.",
    "Patient is a smoker and has a family history of heart disease.",
    "Patient has been diagnosed with chronic kidney disease."
]

# Tokenize and get embeddings
embeddings = []
for text in patient_data:
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())

# Summarize patient data using a summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
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

# Write categorized and summarized patient data to a text file
with open("categorized_patient_data.txt", "w") as file:
    for category, summaries in classified_data.items():
        file.write(f"{category}:\n")
        for summary in summaries:
            file.write(f"• {summary}\n")
        file.write("\n")

print("Categorized patient data has been written to categorized_patient_data.txt")