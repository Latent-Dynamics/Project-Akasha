import torch
from unsloth import FastLanguageModel

# POINTING TO YOUR VERIFIED 1.7B ADAPTERS
ADAPTER_DIR = "titan_dora_adapters"

print("📡 INITIALIZING ANCHOR PROBER: DIMENSIONAL EXTRACTION...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = ADAPTER_DIR,
    load_in_4bit = True,
)

def probe_activations(prompt):
    print(f"🔍 Probing Latent Space for: {prompt[:50]}...")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # We use a hook to capture the 'Hidden States' (the actual neurons firing)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        
    # Layer 28 is the 'Final Thought' layer. We extract its 2048-dim vector.
    final_layer_state = outputs.hidden_states[-1]
    
    # Calculate 'Density Score' (Mean Absolute Activation)
    density_score = torch.abs(final_layer_state).mean().item()
    print(f"📊 Manifold Density Score: {density_score:.4f}")
    
    # Return the coordinates of the strongest 1% of neurons (The Anchors)
    top_neurons = torch.topk(final_layer_state.flatten(), k=20).indices
    return top_neurons

if __name__ == "__main__":
    # TASK: The 1-Trillion Parameter Hardware Test
    test_task = "Write a PyTorch kernel for 4-bit dequantization in a single CUDA stream with shared memory offsets."
    
    anchors = probe_activations(test_task)
    print(f"📍 FOUND ANCHOR POINTS (Top Neuron Indices): {anchors.tolist()}")
    
    print("\n[NEXT STEP]: We must freeze these indices into a 'Static Bias Map'.")
