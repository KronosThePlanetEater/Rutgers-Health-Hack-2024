import os
from transformers import MarianMTModel, MarianTokenizer

def translate_text(text, src_lang="en", tgt_lang="es"):
    """
    Translate text from source language to target language using MarianMT.
    
    :param text: List of strings to translate.
    :param src_lang: Source language code.
    :param tgt_lang: Target language code.
    :return: List of translated strings.
    """
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    translated = []
    for t in text:
        translated_text = model.generate(**tokenizer(t, return_tensors="pt", padding=True))
        translated.append(tokenizer.decode(translated_text[0], skip_special_tokens=True))
    
    return translated

def main():
    # Prompt for the file path
    file_path = input("Enter the path to the summarized text file (.txt or .md): ")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return
    
    # Check if the file extension is .txt or .md
    if not (file_path.endswith('.txt') or file_path.endswith('.md')):
        print("File must be a .txt or .md file.")
        return
    
    # Read summarized text from file
    with open(file_path, 'r', encoding='utf-8') as file:
        summarized_text = file.readlines()
    
    # Translate summarized text
    translated_text = translate_text(summarized_text, src_lang="en", tgt_lang="es")
    
    # Save translated text to a new file
    translated_file_path = "translated_text.txt"
    with open(translated_file_path, 'w', encoding='utf-8') as file:
        for line in translated_text:
            file.write(line + "\n")
    
    print(f"Translated text saved to {translated_file_path}")

if __name__ == "__main__":
    main()