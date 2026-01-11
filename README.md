# Latent-Dynamics: Project Akasha

**Status:** Active Research
**Architecture:** Asymmetric Skeleton Steering

## Research Objective
To enable high-fidelity reasoning on consumer hardware by decoupling "World Knowledge" (Latent Geometry) from "Active Inference" (Token Generation).

## System Topology
We utilize a split-stack architecture to maximize available memory bandwidth:

### 1. The Active Worker (1.7B Parameters)
* **Role:** Active Inference & Token Generation.
* **Location:** GPU VRAM.
* **Function:** Rapidly processes raw data streams ("Mud") to generate candidate logic.

### 2. The Skeleton Map (32B Parameter Structure)
* **Role:** Static Reference & Geometry.
* **Location:** System RAM (Offloaded/Soft-Loaded).
* **Function:** The 32B model is not active. It serves as a frozen coordinate system. The Worker model's latent vectors are projected against this "Skeleton" to correct drift and verify logical coherence.

### 3. The Bridge
* **Technique:** Latent Memory Projectors using DoRA (Weight-Decomposed Low-Rank Adaptation) and State Space Models (SSM).
* **Goal:** To allow the 1.7B model to "navigate" using the map of the 32B model.

## License
**GNU AGPLv3**
Reciprocity required. Modifications must be open-sourced.

## Architecture Diagram

```mermaid
graph TD
    A[Raw Data 'Mud'] -->|Input| B(1.7B Active Worker / GPU)
    B -->|Generates Vectors| C{Latent Projector}
    D[32B Static Skeleton / RAM] -.->|Reference Geometry| C
    C -->|Corrected Logic| E[Diamond Axioms]
    style B fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
