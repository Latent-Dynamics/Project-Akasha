# Asymmetric Skeleton Steering & Recursive Code Evolution

**Author:** Latent-Dynamics Research
**Date:** January 2026
**Status:** Active Methodology

## 1. Abstract
This report details a novel architecture for local AI inference that decouples "World Knowledge" from "Active Generation." By utilizing a **Static Skeleton Strategy**, we maintain the weight structure of a 32B parameter model in system RAM as a passive reference map, while a high-speed 1.7B parameter model performs active inference on the GPU. Furthermore, we introduce an "Evolutionary Coding" pipeline where the larger model acts as a critic for the smaller model's output, enabling autonomous code optimization.

## 2. Methodology: The "Skeleton Steering" Protocol

### 2.1 The Problem
Small Language Models (SLMs) possess high speed but suffer from "hallucination" due to compressed latent spaces. Large models (30B+) are accurate but computationally prohibitive for high-volume iteration on consumer hardware.

### 2.2 The Solution: Soft-Loaded Guidance
Instead of running full inference on the 32B model, Project Akasha "soft-loads" its attention heads and vector topology into system RAM.
* **The Worker (1.7B):** Generates tokens rapidly on the GPU.
* **The Skeleton (32B):** Does not generate text. It acts as a "Gravity Well," correcting the vector trajectory of the Worker model to ensure logical consistency.

## 3. Recursive Code Evolution (The "Mud" Loop)

### 3.1 Concept
Beyond simple inference, the system utilizes a Darwinian selection mechanism for code generation. We posit that "Tinies" (Specialist SLMs) provide high-variance creative output ("The Mud"), while the "Skeleton" provides high-stability logical verification ("The Diamond").

### 3.2 The Evolutionary Cycle
1.  **Divergent Generation:** The Specialist models (running on CPU/RAM) generate multiple raw variations of a code function.
2.  **Latent Critique:** The 32B Skeleton scores these variations based on logical coherence and structural integrity.
3.  **Convergent Refinement:** The system discards the "hallucinated" code and iteratively refines the winning logic.
4.  **Result:** Autonomous discovery of optimized algorithms that neither model could produce individually.

## 4. Architecture Diagram

```mermaid
graph TD
    A[Task Request] --> B(1.7B Worker / GPU)
    B -->|Generates Variants| C[Raw 'Mud' Pool]
    D[32B Skeleton / RAM] -.->|Latent Guidance| B
    C -->|Projected to| E{Skeleton Critique}
    E -->|Rejects| F[Trash]
    E -->|Selects| G[Optimized 'Diamond' Code]
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
