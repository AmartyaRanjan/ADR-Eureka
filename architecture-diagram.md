# System Architecture — Physics-Informed Adaptive Reward Generation

Two feedback loops share one diagnosis router: the inherited Eureka reward-edit loop, and the new adaptive-DR loop. The router is the safeguard that keeps the PINN's contribution measurable — each diagnosis can only trigger its own loop.

```mermaid
flowchart TD
    TASK["Task spec + safety instruction<br/>+ environment source code"]

    subgraph GEN["Reward Generation Loop — Eureka (inherited)"]
        LLM["Coding LLM<br/>reward generation<br/>K candidates, evolutionary search"]
        TRAIN["Parallel PPO Training<br/>Isaac Gym — Shadow Hand / LEAP Hand"]
    end

    subgraph DIAG["Diagnosis Layer (new)"]
        PINN["Physics-Informed Network<br/>residual loss: EOM + soft contact<br/>→ consistency signal"]
        SUMM["Failure-Mode Summarizer<br/>MLP → outcome + failure phase"]
        ROUTER{"Diagnosis Router (LLM)<br/>reward-logic vs. physics mismatch"}
    end

    subgraph DR["Adaptive DR Loop (new)"]
        ADR["Adaptive DR Controller<br/>widen / narrow ranges every N iters"]
        SAMPLER["DR Sampler φ"]
    end

    TASK --> LLM
    LLM -->|K reward candidates| TRAIN
    SAMPLER -->|current ranges| TRAIN
    TRAIN -->|telemetry: joints, contact, DR params| PINN
    TRAIN -->|telemetry + fitness F| SUMM
    PINN -->|consistency signal| ROUTER
    SUMM -->|failure descriptors| ROUTER
    ROUTER -->|reward-logic error| LLM
    ROUTER -->|physics mismatch| ADR
    ADR -->|new ranges φ'| SAMPLER

    classDef hot fill:#F3E2D4,stroke:#C05A1D,stroke-width:2px,color:#17282F;
    classDef cool fill:#DCEAE7,stroke:#186660,stroke-width:2px,color:#17282F;
    classDef neutral fill:#F5F8F7,stroke:#8A9490,stroke-width:1.5px,color:#17282F;

    class LLM,ROUTER hot;
    class ADR,SAMPLER cool;
    class TRAIN,PINN,SUMM,TASK neutral;
```

**Legend**
- Orange nodes — reward-edit path (unmodified from Eureka)
- Teal nodes — adaptive-DR path (new contribution)
- Grey nodes — shared infrastructure (training loop, diagnosis inputs)

---

## Simplified linear version (for the flow-diagram slide)

```mermaid
flowchart LR
    A["Dataset<br/>simulated hand + object rollouts"] --> B["Preprocessing<br/>reward reflection + PINN diagnosis signal"]
    B --> C["Model<br/>LLM reward generator + adaptive DR + PINN router"]
    C --> D["Training<br/>PPO, Isaac Gym"]
    D --> E["Output<br/>validated reward + DR config + trained policy"]
    E -.->|next iteration| B
```
