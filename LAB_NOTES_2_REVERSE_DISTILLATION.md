# Protocol: Hierarchical Axiom Synthesis ("Reverse Distillation")

**Status:** Proposed Architecture
**Date:** January 11, 2026

## 1. Hypothesis
Standard industry practice uses "Knowledge Distillation" to compress large models into smaller ones (Teacher $\rightarrow$ Student). This results in lossy compression.
**We propose the inverse:** A multi-stage filtration pipeline where low-parameter models generate high-variance data ("Mud"), which is progressively refined by increasingly complex models to synthesize a high-density training dataset.

## 2. The "Ladder" Topology
Data flows upwards through four distinct filtration layers.

1.  **Generator Layer (The Mud):** 0.5B Parameter Models (CPU). High temperature, maximum variance.
2.  **Structuring Layer (The Worker):** 1.7B Parameter Model (GPU). Formats raw output into logical syntax.
3.  **Verification Layer (The Critic):** 8B Parameter Model (VRAM). Checks against known axioms.
4.  **Reference Layer (The Skeleton):** 32B Static Map (RAM). Final latent alignment check.

## 3. Implementation Logic (Pseudocode)

```python
class AxiomLadder:
    def __init__(self):
        self.generator = TinyModel(device='cpu')   # 0.5B
        self.worker = WorkerModel(device='gpu')    # 1.7B
        self.critic = RefinerModel(device='vram')  # 8B
        self.skeleton = StaticMap(device='ram')    # 32B (Frozen)

    def refine_axiom(self, raw_prompt):
        # Stage 1: Generate high-variance candidates (The Mud)
        candidates = self.generator.generate_batch(raw_prompt, n=50)

        # Stage 2: Structure and Syntax Check
        structured = []
        for c in candidates:
            if self.worker.check_syntax(c):
                structured.append(c)

        # Stage 3: Logical Verification
        diamonds = []
        for s in structured:
            score = self.critic.verify_logic(s)
            if score > 0.95: # Strict Threshold
                diamonds.append(s)

        # Stage 4: Skeleton Alignment (Latent Check)
        final_dataset = []
        for d in diamonds:
            if self.skeleton.validate_vector_path(d):
                final_dataset.append(d)

        return final_dataset # The Golden Dataset

        '''

graph BT
    A[Raw Data Input] --> B(0.5B Generator / CPU)
    B -->|High Variance Batch| C(1.7B Worker / GPU)
    C -->|Structured Logic| D(8B Critic / VRAM)
    D -->|Verified Axioms| E[32B Skeleton / RAM]
    E -->|Golden Data| F[High-Density Target Model]
    style F fill:#f9f,stroke:#333,stroke-width:2px

    License: GNU AGPLv3
