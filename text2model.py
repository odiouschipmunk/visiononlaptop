import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def read_frame_analysis(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def process_prompt_with_model(prompt):
    # Load the model and tokenizer
    model_name = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Encode the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text
    outputs = model.generate(**inputs, max_new_tokens=4096, num_return_sequences=1)
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    frame_analysis_path = 'output/frame_analysis.txt'
    prompt = 'you are a squash coach, analyze this data and tell me exactly what is happening: ' + read_frame_analysis(frame_analysis_path)

    
    response = process_prompt_with_model(prompt)
    print(response)