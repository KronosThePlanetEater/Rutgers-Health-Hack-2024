from transformers import AutoTokenizer, AutoModel

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

# Example usage
text = "Patient has a history of diabetes and hypertension."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print(outputs)