import torch

def read_frame_analysis(file_path):
    with open(file_path, 'r') as file:
        return file.read()

from transformers import GPTNeoForCausalLM, GPT2Tokenizer

def process_prompt_with_model(prompt):
    # Load the model and tokenizer
    model_name = 'EleutherAI/gpt-neo-1.3B'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPTNeoForCausalLM.from_pretrained(model_name)

    # Set pad_token_id to eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True).to(device)

    # Generate text
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        num_return_sequences=1
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

if __name__ == "__main__":
    frame_analysis_path = 'output/frame_analysis.txt'
    prompt = 'you are a squash coach, analyze this data and tell me exactly what is happening: ' + read_frame_analysis(frame_analysis_path)
    
    # Ensure the prompt is not too long
    if len(prompt) > 1024:
        prompt = prompt[:1024]
    
    response = process_prompt_with_model(prompt)
    print(response)