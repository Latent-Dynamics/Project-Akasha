import torch
import gc
import json
from unsloth import FastLanguageModel

# ==================================================
# ☢️  PROJECT AKASHA: GODZILLA-TIER CORE ENGINE
# Specialized for Qwen 3 1.7B | RTX 4080 Alignment
# ==================================================

# Pointing to your verified local directory
ADAPTER_DIR = "titan_dora_adapters"

class AkashaGodzilla:
    def __init__(self):
        print("\n" + "="*50)
        print("☢️  INITIALIZING GODZILLA-TIER SPECIALIST SQUAD")
        print("="*50)
        
        # Load Roles from your JSON registry
        try:
            with open('roles.json', 'r') as f:
                self.roles = json.load(f)
            print("[✔] God-Tier Roles Loaded from Registry.")
        except FileNotFoundError:
            print("[!] CRITICAL ERROR: roles.json not found. Creating emergency map...")
            self.roles = {"builder": "ERROR", "architect": "ERROR", "scholar": "ERROR"}

        # Load the base model (Corrected for 1.7B / 2048-dim alignment)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = ADAPTER_DIR, 
            load_in_4bit = True,
        )

        # Pre-Load Specialty Adapters
        print("[+] Welding Logic Anchors: Architect, Builder, Scholar...")
        self.model.load_adapter(ADAPTER_DIR, adapter_name="architect")
        self.model.load_adapter(ADAPTER_DIR, adapter_name="builder")
        self.model.load_adapter(ADAPTER_DIR, adapter_name="scholar")
        print("[✔] System Online. Handshake Success.")

    def run_pipeline(self, user_goal):
        print(f"\n[MISSION START]: {user_goal}")
        
        # --- PHASE 1: THE BUILDER (Code Synthesis) ---
        print("🛠️  BUILDER: Generating Saturated Logic...")
        self.model.set_adapter("builder")
        builder_prompt = f"### ANCHOR: {self.roles['builder']}\nTASK: {user_goal}\nCODE:"
        raw_code = self.infer(builder_prompt)
        self.flush()

        # --- PHASE 2: THE ARCHITECT (Manifold Audit) ---
        print("🔍  ARCHITECT: Performing Hardware-Aware Autopsy...")
        self.model.set_adapter("architect")
        architect_prompt = f"### ANCHOR: {self.roles['architect']}\nAUDIT_TARGET: {raw_code}\nRESULT:"
        audit = self.infer(architect_prompt)
        self.flush()

        # --- PHASE 3: THE SCHOLAR (32B Memory Mapping) ---
        print("🧬  SCHOLAR: Stitching Latent Projections to 32B...")
        self.model.set_adapter("scholar")
        scholar_prompt = f"### ANCHOR: {self.roles['scholar']}\nALIGN_CODE: {raw_code}\nMAP:"
        mapping = self.infer(scholar_prompt)
        self.flush()

        return raw_code, audit, mapping

    def infer(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # GODZILLA-TIER GENERATION PARAMS (Prevents Loops)
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=400,
            temperature=0.1,           # Low for precision
            repetition_penalty=1.2,    # Prevents "Infinite Manifold-Safe" loops
            no_repeat_ngram_size=5,    # Breaks repetitive sentence structures
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True
        )
        
        # Clean the output to only show the specialist's new thought
        decoded = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        # Cut off any residual hallucinations
        return decoded.split("###")[0].strip()

    def flush(self):
        """Hardware-level VRAM evacuation for the RTX 4080."""
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    akasha = AkashaGodzilla()
    code, audit, mapping = akasha.run_pipeline("Optimize Latent Memory Projectors for PCIe Gen4 offloading.")
    
    print("\n" + "💎 " * 20)
    print("GODZILLA-TIER PIPELINE OUTPUT")
    print("💎 " * 20)
    print(f"\n[1. BUILDER CODE]:\n{code[:500]}...")
    print(f"\n[2. ARCHITECT AUDIT]:\n{audit}")
    print(f"\n[3. SCHOLAR MAPPING]:\n{mapping[:300]}...")
