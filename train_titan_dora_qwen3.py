"""
TRAIN_TITAN_DORA_QWEN3.py - Titan DoRA Fine-Tuning (WSL2 Compatible)
---------------------------------------------------------------------
Fine-tunes Qwen 3 1.7B with DoRA adapters using the Titan dataset.
Designed to run in WSL2 with files copied to ~/akasha_lab/
"""
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import os

# ==============================================================================
# 1. CONFIGURATION (Linux/WSL2 Compatible)
# ==============================================================================
# Paths are relative to current directory (~/akasha_lab/)
# Model will be downloaded from HuggingFace if not local
MODEL_NAME = "Qwen/Qwen3-1.7B"  # HuggingFace model (auto-download)
DATASET_FILE = "./titan_dataset_complete.jsonl"  # Relative path
OUTPUT_DIR = "./titan_dora_adapters"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True  # 4-bit for VRAM efficiency on 1.7B

print("=" * 60)
print("TITAN FORGE: INITIALIZING (WSL2 Mode)")
print("=" * 60)
print(f"[+] Target Model: {MODEL_NAME}")
print(f"[+] Dataset: {DATASET_FILE}")
print(f"[+] Output: {OUTPUT_DIR}")

# Verify dataset exists
if not os.path.exists(DATASET_FILE):
    print(f"[!] ERROR: Dataset not found at {DATASET_FILE}")
    print("    Make sure you copied titan_dataset_complete.jsonl to this folder.")
    exit(1)

# ==============================================================================
# 2. LOAD MODEL
# ==============================================================================
print("\n[+] Loading Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = LOAD_IN_4BIT,
)

# ==============================================================================
# 3. CONFIGURE DoRA
# ==============================================================================
print("[+] Configuring DoRA Adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    use_dora = True,  # ENABLE DoRA
    loftq_config = None,
)

# ==============================================================================
# 4. PREPARE DATA
# ==============================================================================
print("[+] Loading Dataset...")
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
print(f"    Loaded {len(dataset)} examples")

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    
    eos = tokenizer.eos_token 

    for instruction, output in zip(instructions, outputs):
        text = (
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{output}<|im_end|>{eos}" 
        )
        texts.append(text)
    return { "text" : texts }

dataset = dataset.map(formatting_prompts_func, batched = True)

# ==============================================================================
# 5. START TRAINING
# ==============================================================================
print("\n" + "=" * 60)
print("STARTING QWEN 3 1.7B DoRA TRAINING")
print("=" * 60)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR,
    ),
)

trainer.train()

# ==============================================================================
# 6. SAVE ARTIFACTS
# ==============================================================================
print(f"\n[+] Saving DoRA Adapters to '{OUTPUT_DIR}'...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("=" * 60)
print("[SUCCESS] Titan Qwen 3 1.7B DoRA Upgrade Complete!")
print("=" * 60)
