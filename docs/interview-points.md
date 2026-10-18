# Interview points

Short, honest talking points for Jagadish-inference-optimization. Every number quoted is from the
real run shown in the README.

- Batching raises throughput and hurts first-token latency; the table shows both.
- My review found the recommender could pick int4 on cost alone; now it needs eval evidence first.
- I label simulated numbers as simulated; the live runner exists for real ones.
- Prefix caching only pays off when prompts share a long prefix (the repeated-prefix workload).

## Likely follow-up questions

- How would you pick the quality floor? (From the task's tolerance; 0.97 is a starting point, not a rule.)
- Why might int4 be fine for one task and not another? (Reasoning-heavy tasks degrade more.)
- What else changes cost at scale? (Utilisation, which the cost per token hides.)
