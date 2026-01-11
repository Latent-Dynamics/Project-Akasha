import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

# --- TITAN SETUP ---
ADAPTER_DIR = "titan_dora_adapters" 
MAX_SEQ_LENGTH = 2048

class ImpossibleTest:
    def __init__(self):
        print("=== MOUNTING TITAN FOR IMPOSSIBLE GAUNTLET ===")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = ADAPTER_DIR,
            max_seq_length = MAX_SEQ_LENGTH,
            load_in_4bit = True,
        )
        FastLanguageModel.for_inference(self.model)

    def probe(self, title, question):
        print(f"\n\n{'='*50}")
        print(f"🔥 PROBE: {title}")
        print(f"{'='*50}")
        messages = [
            {"role": "system", "content": "You are the Titan Visionary. You synthesize novel, mathematically sound solutions for unsolved engineering paradoxes."},
            {"role": "user", "content": question}
        ]
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        # Higher temperature (0.8) to force the model to 'innovate' rather than just repeat
        self.model.generate(inputs, streamer=streamer, max_new_tokens=1024, temperature=0.8)

if __name__ == "__main__":
    tester = ImpossibleTest()

    # PROBE 1: THE BUS STITCH (Hardware/Software Paradox)
    tester.probe("THE BUS STITCH", 
        "Propose a mathematical way to map VRAM weights to CPU-resident adapters using a 'Latent Bus Projector'. How do we compensate for PCIe latency in the forward pass? Write a theoretical loss function.")

    # PROBE 2: THE RECURSIVE ANCHOR (Mamba + Transformer Bridge)
    tester.probe("THE RECURSIVE ANCHOR", 
        "Propose a tensor operation to inject the Mamba hidden state (h_t) directly into the Transformer's Query (Q) vector. How does this force the model to perceive long-term history as current sensory input?")

    # PROBE 3: NON-EUCLIDEAN OPTIMIZATION (Geometry)
    tester.probe("HYPERBOLIC MUD", 
        "Design a loss function for training on Hyperbolic Geometry (Poincaré disk) to prevent 'Knowledge Drift' in the Mud pipeline. Show the math.")
