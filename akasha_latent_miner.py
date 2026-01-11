import torch
import time
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ==================================================
# ⚡ PROJECT AKASHA: ZERO-TOKEN LATENT MINER (V4)
# Purpose: Vector-Level Code Optimization (Flexible Mode)
# ==================================================

# 1. THE 32B IMPLANTS
MASTER_BLUEPRINTS = [
    # [PASTE YOUR NUMBERS HERE]
]

print("⚡ INITIALIZING FLEXIBLE RIG...")

# We use Standard Transformers + BitsAndBytes for 4-bit loading
# This bypasses the strict checks in Unsloth that block raw vector injection
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model_name = "Qwen/Qwen2.5-1.5B-Instruct" 

print(f"🔌 Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

embedding_layer = model.model.embed_tokens

def sovereign_generate(optimized_latents, max_tokens=512):
    """
    MANUAL TRANSMISSION
    Standard Transformers allows us to pass 'inputs_embeds' 
    without demanding 'input_ids'.
    """
    curr_embeds = optimized_latents.to("cuda")
    generated_ids = []
    past_key_values = None

    with torch.no_grad():
        for i in range(max_tokens):
            # 1. Forward Pass (Pure Vectors)
            # The standard model accepts inputs_embeds=... and input_ids=None
            outputs = model(
                inputs_embeds=curr_embeds,
                past_key_values=past_key_values,
                use_cache=True
            )
            
            # 2. Update Memory
            past_key_values = outputs.past_key_values
            
            # 3. Greedy Decode
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            generated_ids.append(next_token.item())
            
            # 4. Prepare next step
            curr_embeds = embedding_layer(next_token)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)

def latent_swing(task_prompt, iterations=10):
    inputs = tokenizer(task_prompt, return_tensors="pt").to("cuda")
    input_ids = inputs.input_ids
    
    with torch.no_grad():
        current_latents = embedding_layer(input_ids)
    
    # The Refinement Loop
    for i in range(iterations):
        noise = torch.randn_like(current_latents) * 0.05
        current_latents = current_latents + noise

    return sovereign_generate(current_latents)

if __name__ == "__main__":
    task = "Write a Python function to find the longest palindrome in a string. Optimize for O(n)."
    
    print(f"\n🌀 MINING LATENT VEINS: {task}")
    print(f"🔨 Taking Latent Swings (Iterations: 50)...")
    
    start_time = time.time()
    result_code = latent_swing(task, iterations=50)
    end_time = time.time()
    
    print("\n" + "="*50)
    print(f"💎 RESULT COLLAPSED IN {end_time - start_time:.4f}s")
    print("="*50)
    print(result_code)
    print("="*50)
