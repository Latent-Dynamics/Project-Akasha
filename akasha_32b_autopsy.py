import os
import numpy as np
from llama_cpp import Llama

MODEL_PATH = "Qwen3-32B-Q4_K_M.gguf"

print("💀 AWAKENING THE QWEN 3 DEAD BODY...")
# Memory-Mapped to stay within your 32GB RAM limit
llm = Llama(model_path=MODEL_PATH, n_gpu_layers=0, n_ctx=2048, verbose=False, logits_all=True)

def extract_qwen3_plans(task):
    thinking_prompt = f"<|thought|>\n{task}\n<|endofthought|>"
    print(f"🔍 PROBING THINKING MANIFOLD...")
    tokens = llm.tokenize(thinking_prompt.encode('utf-8'))
    llm.eval(tokens)
    
    # Capture the raw output
    logits = np.array(llm._scores)
    # Get the indices of the 100 strongest signals
    top_indices = np.argsort(logits)[-100:][::-1]
    return top_indices.tolist()

if __name__ == "__main__":
    target_task = "Write a PyTorch kernel for 4-bit dequantization. Use __shared__ float tile logic."
    master_indices = extract_qwen3_plans(target_task)
    
    # CLEAN SAVE: Join numbers with commas, NO BRACKETS
    with open("32b_blueprints.txt", "w") as f:
        f.write(",".join(map(str, master_indices)))
    
    print("\n✅ BLUEPRINTS SAVED CLEANLY TO: 32b_blueprints.txt")
    print(f"Sample data: {str(master_indices[:5])}...")
