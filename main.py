import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    print("Loading GPT-OSS 20B model...")
    
    # model_name = "EleutherAI/pythia-160m"
    model_name = "openai/gpt-oss-20b"
    
    print(f"Model: {model_name}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with automatic device placement
    print("Loading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use float16 to reduce memory usage
        device_map="auto",  # Automatically distribute across available devices
        low_cpu_mem_usage=True
    )
    
    print("Model loaded successfully!\n")
    
    # Test with "Hello World" prompt
    prompt = "Hello World! This is a test of"
    
    print(f"Prompt: {prompt}")
    print("-" * 50)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    print("Generating response...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and print result
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated text:\n{generated_text}")
    print("-" * 50)
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()