import torch
import gc
from unsloth import FastLanguageModel

# ==================================================
# 🧬 PROJECT AKASHA: UNIVERSAL ANCHOR MASTER
# One script to Probe, Inject, and Execute.
# ==================================================

ADAPTER_DIR = "titan_dora_adapters"

class AkashaMaster:
    def __init__(self):
        print("\n☢️  AKASHA MASTER: INITIATING HARDWARE HANDSHAKE")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = ADAPTER_DIR,
            load_in_4bit = True,
        )
        print("[✔] 1.7B Specialist Chassis Loaded.")

    def probe_and_lock(self, task_description, strength=5.0):
        print(f"\n📡 STAGE 1: PROBING MANIFOLD FOR ANCHOR POINTS...")
        inputs = self.tokenizer(task_description, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # Extract the high-density neurons from the final layer
            final_layer = outputs.hidden_states[-1]
            # Get the top 20 neuron indices (coordinates)
            top_values, top_indices = torch.topk(final_layer[:, -1, :].flatten(), k=20)
            anchors = top_indices.tolist()
            print(f"📍 ANCHORS DISCOVERED: {anchors}")

        # --- STAGE 2: INJECTION HOOK ---
        print("💉 STAGE 2: INJECTING STATIC BIAS (BACK-DOOR MODE)...")
        
        def steering_hook(module, input, output):
            # Surgical modification of the hidden state hidden_states
            # hidden_states is index 0 of the Unsloth output tuple
            hidden_states = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                for idx in anchors:
                    # Nudge these specific neurons to 'ON'
                    hidden_states[:, -1, idx] += strength
            return output

        # Attach the hook to the final layer automatically
        handle = self.model.get_decoder().layers[-1].register_forward_hook(steering_hook)

        # --- STAGE 3: EXECUTION ---
        print("⚡ STAGE 3: EXECUTING STEERED LOGIC (1% ENERGY)...")
        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.01,
                repetition_penalty=1.2,
                eos_token_id=self.tokenizer.eos_token_id
            )
            result = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print("\n" + "="*50 + "\n💎 TRILLION-PARAMETER RESULT:\n" + "="*50)
            print(result)
        finally:
            handle.remove() # Clean up the brain after use
            self.flush()

    def flush(self):
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # This is the "Trillion Parameter" Complexity Test
    complex_task = (
        "Write a PyTorch kernel for 4-bit dequantization in a single CUDA stream "
        "using shared memory offsets. Use __shared__ float tile logic. No prose."
    )
    
    master = AkashaMaster()
    master.probe_and_lock(complex_task)
