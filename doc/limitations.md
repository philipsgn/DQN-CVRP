# Limitations and Future Research

## Overview

This project demonstrates the application of **Deep Reinforcement Learning (DRL)** to static and semi-dynamic variants of the Capacitated Vehicle Routing Problem.

The experimental results provide evidence that a masked Double DQN can learn meaningful routing behaviour and outperform simpler routing strategies under several conditions.

However, the research also identifies important limitations.

Recognising these limitations is essential when interpreting the results and provides several directions for future research.

---

# 1. Instance-Specific Learning

The current DQN framework trains policies for individual routing instances.

This allows controlled evaluation but limits the ability of the learned policy to generalise directly to completely unseen routing graphs.

A stronger general-purpose routing system would ideally learn structural properties of routing problems rather than memorising behaviour associated with individual benchmark instances.

### Future Direction

Future research could investigate architectures designed for graph-structured problems, including:

- Graph Neural Networks (GNNs);
- Graph Attention Networks;
- attention-based routing models; and
- transformer-based combinatorial optimisation.

These approaches could potentially improve generalisation across different customer configurations and problem sizes.

---

# 2. Limited Dynamic Events

The dynamic environment focuses primarily on changes in travel costs.

This provides a controlled mechanism for studying disruption but represents only one form of uncertainty encountered in real logistics systems.

Real routing environments may experience:

- new customer requests;
- cancelled deliveries;
- road closures;
- vehicle breakdowns;
- changing customer priorities;
- driver availability changes;
- uncertain service times; and
- unexpected delivery delays.

### Future Direction

The environment could be extended to support multiple forms of disruption simultaneously.

This would provide a more realistic test of adaptive routing policies.

---

# 3. High Disruption Remains Challenging

Experimental results show that DQN performance deteriorates as travel-cost disruption becomes stronger.

The advantage over nearest-neighbour routing is statistically clear under static and low-disruption conditions but becomes less clear under high disruption.

This suggests that the current learning architecture does not provide complete robustness to substantial environmental change.

### Future Direction

Future work could investigate:

- robust reinforcement learning;
- distributionally robust optimisation;
- domain randomisation;
- curriculum learning;
- uncertainty-aware policies; and
- adaptive exploration strategies.

---

# 4. Dynamic Training Does Not Guarantee Robustness

Cross-evaluation produced an interesting observation.

Training directly under stronger dynamic conditions did not automatically result in a policy that degraded less when evaluated under disruption.

This suggests that simply exposing a reinforcement-learning agent to greater environmental noise is not sufficient to guarantee robustness.

### Future Direction

A more systematic investigation could examine:

- training-distribution design;
- curriculum-based disruption;
- out-of-distribution evaluation;
- policy calibration;
- uncertainty estimation; and
- explicit robustness objectives.

This represents an important potential research direction beyond the present implementation.

---

# 5. Classical Optimisation Remains Stronger

Google OR-Tools achieves better absolute route quality than the DQN across the principal benchmark comparisons.

This is an important limitation of the learned approach.

The results therefore do not support the claim that Deep Reinforcement Learning should replace classical vehicle-routing optimisation.

Instead, the experiments demonstrate different strengths.

### OR-Tools

Provides strong solution quality through explicit optimisation.

### DQN

Provides a learned sequential policy capable of producing decisions through neural-network inference after training.

### Future Direction

Rather than treating these approaches as competitors, future research could investigate **hybrid optimisation-learning systems**.

For example:

```text
               Routing Problem
                     │
             ┌───────┴───────┐
             ▼               ▼
      Learned Policy    Optimisation
             │               │
             └───────┬───────┘
                     ▼
             Adaptive Decision
```

Possible hybrid strategies include:

- using DRL to warm-start classical optimisation;
- using optimisation solutions as demonstrations;
- using DRL between optimisation cycles;
- invoking optimisation when policy uncertainty becomes high; and
- using learned models to guide classical search.

---

# 6. Generalisation to Unseen Problems

The present evaluation primarily investigates stochastic variation around known benchmark structures.

This differs from demonstrating zero-shot generalisation to entirely unseen routing graphs.

A policy that performs well on a trained instance may not automatically transfer to:

- different customer counts;
- different geographic distributions;
- different vehicle capacities; or
- substantially different routing structures.

### Future Direction

Future evaluation should explicitly separate:

```text
Training Instances
       ↓
Validation Instances
       ↓
Completely Unseen Test Instances
```

This would provide stronger evidence of generalisation.

---

# 7. Scalability

The benchmark collection contains routing problems of different sizes, but the project does not establish scalability to very large industrial routing systems.

As the number of customers increases:

- the action space grows;
- the state representation becomes larger;
- exploration becomes more difficult; and
- reinforcement-learning training becomes more computationally expensive.

### Future Direction

Potential approaches include:

- hierarchical reinforcement learning;
- graph-based state representations;
- attention mechanisms;
- decomposition strategies;
- multi-agent reinforcement learning; and
- hybrid neural-optimisation methods.

---

# 8. Single Main Optimisation Objective

The primary objective considered in this project is route/travel cost.

Real logistics systems frequently optimise several competing objectives.

These may include:

- distance;
- delivery time;
- fuel consumption;
- carbon emissions;
- customer satisfaction;
- driver workload;
- operational cost; and
- service reliability.

### Future Direction

The problem could be reformulated as a **multi-objective reinforcement-learning problem**.

This would allow the routing policy to learn trade-offs between competing operational goals.

---

# 9. Uncertainty Is Simulated Rather Than Predicted

Dynamic travel costs are generated within the experimental environment.

The current system therefore reacts to changing costs but does not attempt to predict future disruption.

In real applications, information could come from:

- traffic sensors;
- weather services;
- historical traffic records;
- GPS traces; or
- real-time road-network data.

### Future Direction

A future system could combine:

```text
Predictive Model
      +
Uncertainty Estimate
      +
Routing Policy
      ↓
Predictive Adaptive Routing
```

For example, a forecasting model could estimate future travel conditions while an RL agent makes routing decisions based on both current observations and predicted uncertainty.

---

# 10. Uncertainty Quantification

The current DQN produces Q-values used for decision making but does not provide a calibrated measure of confidence in those decisions.

This is particularly important in high-stakes or safety-critical AI systems.

A policy may encounter conditions substantially different from those seen during training while still producing a confident action.

### Future Direction

Potential approaches include:

- Bayesian neural networks;
- deep ensembles;
- Monte Carlo dropout;
- distributional reinforcement learning;
- conformal prediction; and
- calibrated uncertainty estimation.

An uncertainty-aware routing system could potentially identify situations where the learned policy should defer to a classical optimisation method.

---

# 11. Distribution Shift

Dynamic routing naturally creates the possibility of **distribution shift**.

A policy trained under one level or type of disruption may encounter substantially different conditions during deployment.

The cross-evaluation results already suggest that training distribution and evaluation distribution can interact in non-trivial ways.

### Future Direction

A more extensive study could explicitly investigate:

> **How reliable is a learned routing policy when deployed under conditions that differ from its training distribution?**

This could involve:

- controlled out-of-distribution benchmarks;
- adversarial disruption;
- unseen traffic patterns;
- unseen graph structures; and
- uncertainty-aware failure detection.

---

# 12. Real-World Validation

CVRPLIB provides an established benchmark environment and is appropriate for controlled algorithmic comparison.

However, benchmark performance does not automatically translate into real-world logistics performance.

Real-world systems include operational constraints that are difficult to represent completely in synthetic or benchmark environments.

### Future Direction

A stronger validation stage could use:

- real delivery data;
- road-network travel times;
- GPS trajectories;
- historical traffic patterns; and
- operational constraints supplied by logistics organisations.

A digital-twin environment could also provide an intermediate step between benchmark evaluation and real deployment.

---

# 13. Computational Cost of Training

Although neural inference can be fast after training, reinforcement-learning training itself can be computationally expensive.

The complete computational comparison should therefore distinguish between:

```text
Training Cost
     versus
Decision-Time Cost
```

A learned policy may require substantial offline training but provide rapid online inference.

Classical optimisation may require little or no training but incur computational cost each time a new solution is calculated.

### Future Direction

Future research should evaluate total computational cost more comprehensively, including:

- training time;
- energy consumption;
- inference latency;
- optimisation latency; and
- frequency of required re-planning.

---

# 14. Research Opportunities

The limitations identified above suggest several promising extensions.

Particularly interesting directions include:

### Graph-Based Deep Reinforcement Learning

Using GNNs or attention mechanisms to improve generalisation across routing graphs.

### Robust Reinforcement Learning

Developing policies that remain reliable under stronger distribution shift.

### Uncertainty-Aware Routing

Estimating when a learned policy is confident enough to act autonomously.

### Hybrid AI and Optimisation

Combining the speed of learned inference with the solution quality and constraints of classical optimisation.

### Multi-Objective Routing

Balancing distance, cost, emissions, reliability, and other operational objectives.

### Real-Time Dynamic Routing

Extending the framework to new orders, cancellations, road closures, and other events occurring during route execution.

---

# 15. Proposed Future Research Architecture

A possible extension of this project could combine:

```text
          Real-Time Environment
                  │
                  ▼
       Graph State Representation
                  │
                  ▼
        Deep RL Routing Policy
                  │
           ┌──────┴──────┐
           ▼             ▼
     Uncertainty     Optimisation
      Estimation       Engine
           │             │
           └──────┬──────┘
                  ▼
        Trusted Routing Decision
                  │
                  ▼
          Environment Update
```

Such a framework would move beyond simply asking:

> "Can reinforcement learning solve a routing problem?"

toward the more challenging research question:

> **"How can learned and optimisation-based decision systems be combined to make fast, robust and trustworthy decisions under uncertainty and distribution shift?"**

---

# Conclusion

The present research demonstrates a working Deep Reinforcement Learning framework for studying dynamic vehicle route optimisation.

It also identifies clear limitations relating to:

- solution quality;
- robustness;
- generalisation;
- scalability;
- uncertainty;
- realism; and
- training cost.

These limitations do not invalidate the experimental findings.

Instead, they define the boundary of the current contribution and provide clear opportunities for further research.

A particularly promising direction is the development of **hybrid, uncertainty-aware AI systems that combine learned sequential decision-making with classical optimisation**.

---

## Related Documentation

For the research approach:

**[Research Methodology](methodology.md)**

For the benchmark and evaluation design:

**[Experimental Design](experiments.md)**

For the empirical findings:

**[Results and Statistical Analysis](results.md)**

For instructions to reproduce the experiments:

**[Reproducibility Guide](reproducibility.md)**

Return to:

**[Project README](../README.md)**