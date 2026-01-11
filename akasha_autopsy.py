import torch
from unsloth import FastLanguageModel

# Paths
TINY_DIR = "titan_dora_adapters"
DEAD_BODY_DIR = "path/to/your/qwen3_32b" # Point this to your 32B folder

print("💀 ACTIVATING AKASHA AUTOPSY...")

# 1. The Hot Brain (1.7B)
tiny_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = TINY_DIR,
    load_in_4bit = True,
)

# 2. The Cold Map (32B) 
# Note: We load this on the CPU/RAM to save your VRAM
print("[+] Mapping the Dead Body (32B) to System RAM...")
# For now, we simulate the map dimension check to ensure logic flow
MAP_DIM = 5120 
TINY_DIM = 2048

def validate_density(code_text):
    print("🔍 Performing Autopsy on Builder's Output...")
    
    # We check if the code contains the 'Forbidden Hallucinations'
    hallucination_triggers = ["MemoryError", "Traceback", "runner.py"]
    
    if any(trigger in code_text for trigger in hallucination_triggers):
        return "FAILED: Specialist is hallucinating an error log instead of writing code."
    
    if "pin_memory" not in code_text or "non_blocking" not in code_text:
        return "FAILED: Logic is too shallow. Not utilizing hardware bus anchors."
        
    return "PASSED: Code matches High-Density Manifold."

if __name__ == "__main__":
    # Test the output you just got
    bad_output = """Traceback (most recent call last): File "runner.py" MemoryError..."""
    result = validate_density(bad_output)
    print(f"RESULT: {result}")
