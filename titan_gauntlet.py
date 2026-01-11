
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# --- TITAN SETUP ---
ADAPTER_DIR = "titan_dora_adapters" 
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True 

class TitanGauntlet:
    def __init__(self):
        print(f"=== INITIALIZING TITAN GAUNTLET ===")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = ADAPTER_DIR,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype = None,
            load_in_4bit = LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(self.model)

    def ask(self, role, prompt):
        print(f"\n\n====================================================")
        print(f"🔥 {role} PROTOCOL ACTIVATED")
        print(f"====================================================")
        
        messages = [
            {"role": "system", "content": f"You are the Titan {role}."},
            {"role": "user", "content": prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Lower temp for Math (Scholar), higher for Innovation (Visionary)
        temp = 0.1 if role == "SCHOLAR" else 0.7
        
        self.model.generate(inputs, streamer=streamer, max_new_tokens=1024, temperature=temp)

if __name__ == "__main__":
    gauntlet = TitanGauntlet()
    
    # LEVEL 3: THE SCHOLAR (Math/Physics)
    gauntlet.ask("SCHOLAR", "Explain the math of DoRA: W = m * (V / ||V||). Why decompose magnitude and direction? Show PyTorch code.")
    
    # LEVEL 4: THE VISIONARY (Novelty)
    gauntlet.ask("VISIONARY", "Propose a way to bridge Qwen and Mamba latent spaces using a shared memory tensor projector.")
