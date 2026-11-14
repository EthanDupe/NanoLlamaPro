# NanoLlamaPro

Autonomous Self-Training Causal Transformer LLM (PyTorch GPU/CPU + Vectorized NumPy fallback) with:
- Grouped-Query Attention (GQA), RoPE, Pre-RMSNorm, SwiGLU, KV-Cache, Min-P sampling
- Fast incremental BPE tokenizer & multi-sentence paragraph packing
- Broad English-only web crawling (Wikipedia, Simple Wikipedia, Reddit RSS, 18 RSS feeds, ArXiv)
- Loss-masked multi-turn & `<THINK>` SFT + fact-grounded Self-Play & GRPO RL
- Automatic Wi-Fi/LAN multi-PC distributed training (FedAvg)

## Quick Start
1. Double-click `launch_nanollama.py` (or `self_train.py`) to auto-install dependencies and start training.
2. Or run `python self_train.py --forever` from a terminal.
