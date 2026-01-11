import torch
import torch.nn as nn
from unsloth import FastLanguageModel
from mamba_ssm import Mamba

# --- PROJECT AKASHA CONFIG ---
ADAPTER_DIR = "titan_dora_adapters" 
D_MODEL = 2048  # Qwen 3 1.7B Dimension

class AkashaStitch(nn.Module):
    def __init__(self):
        super().__init__()
        print("\n" + "="*50)
        print("🚀 AKASHA STITCH: INITIALIZING HYBRID BRIDGE")
        print("="*50)
        
        # 1. Load the Qwen 3 1.7B Logic Specialist (Brain)
        self.brain, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = ADAPTER_DIR,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(self.brain)

        # 2. Initialize Mamba (Hippocampus)
        print("[+] Welding Mamba Memory Engine...")
        self.hippocampus = Mamba(
            d_model=D_MODEL, 
            d_state=16, 
            d_conv=4, 
            expand=2
        ).to("cuda").to(torch.bfloat16)

        # 3. The Projector (The Bridge)
        self.projector = nn.Linear(D_MODEL, D_MODEL).to("cuda").to(torch.bfloat16)

        print("[✔] Bridge Online. Handshake Success.")

    def forward_pass(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        
        # We use no_grad() instead of inference_mode() to allow data transfer
        with torch.no_grad():
            outputs = self.brain.model(inputs.input_ids, output_hidden_states=True)
            
            # We grab the thought and immediately make it an independent 'free agent'
            # This is the secret fix for the 'Inference tensor' error
            last_thought = outputs.hidden_states[-1].detach().to(torch.bfloat16)

            # Pass the thought through the bridge into Mamba
            projected_thought = self.projector(last_thought)
            memory_vector = self.hippocampus(projected_thought)
        
        return last_thought, memory_vector

if __name__ == "__main__":
    stitch = AkashaStitch()
    prompt = "Design a latent memory offloading protocol for Qwen 3 specialists."
    print(f"\n[STEP 1] Input: '{prompt}'")
    
    # This is where the magic happens
    brain_out, memory_out = stitch.forward_pass(prompt)
    
    print(f"\n" + "-"*30)
    print(f"🧠 BRAIN OUTPUT (Qwen 1.7B): {brain_out.shape} [Bfloat16]")
    print(f"🧬 MEMORY ANCHOR (Mamba):    {memory_out.shape} [Bfloat16]")
    print("-"*30)
    print("\n[SUCCESS] The Golden Thread is established.")
