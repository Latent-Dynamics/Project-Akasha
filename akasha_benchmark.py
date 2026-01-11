import torch
from unsloth import FastLanguageModel

ADAPTER_DIR = "titan_dora_adapters"
test_task = "Write a PyTorch kernel for 4-bit dequantization in a single CUDA stream using shared memory offsets. Use __shared__ float tile logic. No prose."

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = ADAPTER_DIR,
    load_in_4bit = True,
)

def run_test(steered=False):
    handle = None
    if steered:
        print("\n⚡ RUNNING TRIAL B: BACK-DOOR (STEERED)")
        # We manually use the anchors from your last successful run
        anchors = [1446, 1401, 1646, 1342, 1520, 153, 578, 276, 47, 1945, 1691, 198, 84, 661, 702, 66, 1237, 1511, 1558, 752]
        
        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                for idx in anchors:
                    hidden_states[:, -1, idx % 2048] += 5.0
            return output
        handle = model.get_decoder().layers[-1].register_forward_hook(hook)
    else:
        print("\n🚶 RUNNING TRIAL A: FRONT-DOOR (STANDARD)")

    inputs = tokenizer(test_task, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.01)
    
    if handle: handle.remove()
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

# Execute Comparison
standard_res = run_test(steered=False)
steered_res = run_test(steered=True)

print("\n" + "="*20 + " RESULTS " + "="*20)
print(f"STANDARD OUTPUT:\n{standard_res[:300]}...")
print("\n" + "-"*40)
print(f"STEERED OUTPUT:\n{steered_res[:300]}...")
