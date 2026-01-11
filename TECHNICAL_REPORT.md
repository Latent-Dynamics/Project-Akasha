# Methodology: Static Skeleton Guidance for SLMs

**Abstract**
This report details a method for improving Small Language Model (SLM) coherence. By maintaining the weight structure of a 32B parameter model in system RAM as a passive reference ("The Skeleton"), we can steer the latent trajectory of an active 1.7B parameter model.

**Hypothesis**
A small model has sufficient vocabulary but insufficient "world geometry." By projecting its hidden states against a larger, frozen model's topology, we can reduce hallucination without the computational cost of full 32B inference.

**Current Status**
- **Worker:** 1.7B (Qwen/Llama based)
- **Reference:** 32B Skeleton
- **Projector:** Custom DoRA implementation
