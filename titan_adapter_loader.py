import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# --- TITAN CONFIGURATION ---
# We point to the ADAPTER directory. Unsloth is smart enough to find the base model automatically.
ADAPTER_DIR = "titan_dora_adapters" 
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True 

class TitanEngine:
    def __init__(self):
        print(f"=== TITAN ENGINE: ONLINE ===")
        print(f"[+] Mounting Titan Adapter from: {ADAPTER_DIR}...")
        
        # 1. Load the Adapter & Base Model
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = ADAPTER_DIR, 
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = None,
            load_in_4bit = LOAD_IN_4BIT,
        )
        
        # Enable 2x faster inference
        FastLanguageModel.for_inference(self.model)
        print("[+] Titan Adapter Mounted. System Ready.")

    def engage(self, role: str, instruction: str):
        # The Role Prompts (The "Emergent Highway")
        role_prompts = {
            "ARCHITECT": "You are the Titan Architect. Focus on DDD, Systems Design, and Patterns.",
            "BUILDER": "You are the Titan Builder. Write production-grade, optimized Python code.",
            "SHIELD": "You are the Titan Shield. Audit for security, OWASP, and vulnerabilities.",
            "DEFAULT": "You are a helpful AI assistant."
        }
        
        system_prompt = role_prompts.get(role, role_prompts["DEFAULT"])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")

        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        print(f"\n--- [{role} ACTIVATED] ---")
        self.model.generate(
            inputs,
            streamer = streamer,
            max_new_tokens = 512,
            use_cache = True,
            temperature = 0.1,
        )
        print("\n--- [END TRANSMISSION] ---")

if __name__ == "__main__":
    engine = TitanEngine()
    
    # TEST 1: The Architect (High Level)
    print("\n[TEST 1] Asking Architect for a Blueprint...")
    engine.engage("ARCHITECT", "Design a high-frequency trading bot architecture using Python and AsyncIO.")
    
    # TEST 2: The Builder (Low Level Code)
    print("\n[TEST 2] Asking Builder for Code...")
    engine.engage("BUILDER", "Write the AsyncIO connection handler for the Binance Websocket API based on that design.")
