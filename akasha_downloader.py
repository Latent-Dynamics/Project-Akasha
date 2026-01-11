from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen3-32B-GGUF"
FILENAME = "Qwen3-32B-Q4_K_M.gguf"

print(f"🚀 INITIATING DOWNLOAD: {FILENAME} from {REPO_ID}")
print("This will bypass CLI path errors. Please wait...")

try:
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        local_dir=".",
        local_dir_use_symlinks=False
    )
    print(f"\n✅ DOWNLOAD COMPLETE!")
    print(f"File located at: {path}")
except Exception as e:
    print(f"\n❌ DOWNLOAD FAILED: {e}")
