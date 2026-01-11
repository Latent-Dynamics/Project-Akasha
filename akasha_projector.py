import os
import re

# 2026 Qwen 3 ARCHITECTURE SPECS
W_32B = 5120
W_1_7B = 2048

def project_anchors():
    if not os.path.exists("32b_blueprints.txt"):
        print("❌ ERROR: 32b_blueprints.txt not found. Run autopsy first.")
        return

    with open("32b_blueprints.txt", "r") as f:
        raw_data = f.read()
        
    # REGEX CLEANING: Find all sequences of digits, ignoring [ ] and other symbols
    clean_indices = re.findall(r'\d+', raw_data)
    indices_32b = [int(i) for i in clean_indices]

    print(f"🛰️  Processing {len(indices_32b)} server-tier anchors...")

    # Dimensional Remapping Math: 5120 -> 2048
    projected = []
    for idx in indices_32b:
        # Scale the address from the 32B map to the 1.7B map
        new_idx = int(round(idx * (W_1_7B / W_32B)))
        if new_idx not in projected:
            projected.append(new_idx)
            
    return projected[:40]

if __name__ == "__main__":
    local_blueprints = project_anchors()
    if local_blueprints:
        print("\n" + "="*50)
        print("🏗️  PROJECTED 1.7B CONSTRUCTION PLAN (THE 95%):")
        print("="*50)
        print(local_blueprints)
        print("="*50)
        print("\n[SUCCESS]: Copy these 40 local coordinates into your master loop.")
