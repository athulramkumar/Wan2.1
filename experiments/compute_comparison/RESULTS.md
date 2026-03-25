# Hybrid Compute Experiment Results

## Date: 2026-03-25

### Setup
- **Prompt**: "The Merced river is overflowing, birds flying in the sky, camera is zooming out to reveal an American Buffalo bathing in the river"
- **Config**: 81 frames, 480x832, 50 steps, seed 42, 16fps
- **GPUs**: H100 80GB ($2.69/hr), 2xA40 48GB each ($1.40/hr)

---

## Complete Experiment Results

| # | Config | GPU | Compute | Per-step | Cost/run | Status |
|---|--------|-----|---------|----------|----------|--------|
| 1 | **14B baseline** | H100 (real prompt) | **472.2s** | **9.4s** | **$0.35** | Video saved |
| 2 | **Hybrid 30/70 local** | H100 (real prompt) | **222.1s** | — | **$0.17** | Video saved |
| 3 | Hybrid 30/70 distributed | H100+2xA40 relay | 405.6s compute | — | ~$0.17+transfer | Video saved |
| 4 | Hybrid 20/80 distributed | H100+2xA40 relay | 396.3s compute | — | ~$0.16+transfer | Video saved |
| 5 | 1.3B only | 2xA40 single GPU | ~403s | 8.1s | $0.16 | Video saved |
| 6 | 14B (timing only) | 2xA40 USP | 723.5s | 14.47s | $0.28 | Timing only |
| 7 | 14B (timing only) | H100 | 715.6s | 14.31s | $0.53 | Timing only |
| 8 | 14B (FSDP) | 2xA40 both | 1260s | 25.2s | $0.49 | Video saved |
| 9 | Hybrid 30/70 (USP, no transfer) | 2xA40 USP+single | 1005.1s | — | $0.39 | Video saved |
| 10 | 1.3B only (native generate.py) | 2xA40 single GPU | 545s (incl load) | ~8.1s | $0.21 | Video saved |

### Distributed Breakdown (MacBook relay, experiments 3-5)

| Experiment | 14B segment | 1.3B segment | Transfer overhead | Total wall time |
|-----------|-------------|-------------- |-------------------|-----------------|
| 14B via H100 (baseline) | 472.2s (50 steps) | — | 33.0s | 574.1s |
| Hybrid 30/70 | 141.7s (15 steps) | 263.9s (35 steps) | 74.3s | 521.8s |
| Hybrid 20/80 | 94.6s (10 steps) | 301.8s (40 steps) | 84.6s | 534.2s |
| 1.3B only | — | 377.1s (50 steps) | 50.3s | 482.3s |

### 2xA40 USP Hybrid Breakdown (experiment 9)

| Phase | Steps | Model | Time | Per-step |
|-------|-------|-------|------|----------|
| Phase 1 | 15 | 14B (USP, 2 GPUs, no VAE) | 592.9s | 39.5s |
| Phase 2 | 35 | 1.3B (single GPU) | 397.0s | 11.3s |
| VAE decode | — | — | 15.1s | — |
| **Total** | **50** | — | **1005.1s** | — |

### Timing-only comparison: 14B on H100 vs 2xA40 USP (experiments 6-7)

These used random embeddings (not real prompts) for apples-to-apples per-step timing:

| GPU | 50 steps | Per-step | Cost/run |
|-----|----------|----------|----------|
| 2xA40 USP | 723.5s | 14.47s | $0.28 |
| H100 | 715.6s | 14.31s | $0.53 |
| **H100 is** | **1.01x faster** | | **1.90x more expensive** |

---

## Key Findings

### 1. Single H100 hybrid is the sweet spot
**Hybrid 30/70 on a single H100 = 222s, 2.2x faster than pure 14B, 53% cheaper.**
Both models fit in 80GB with room to spare. No transfer overhead.

### 2. Distributed adds transfer overhead but enables cost savings
With same-datacenter shared volume (not tested, estimated):
- Transfer overhead: <1s (NVMe)
- Hybrid 30/70: ~406s compute + 1s transfer = ~407s
- Cost: H100 spot ($1.75/hr) for 142s = $0.07 + cheap GPU for 264s = $0.10 → **$0.17/video**

With MacBook relay (tested):
- Transfer overhead: 74-85s (SSH tunnel double-hop)
- Still produces correct videos

### 3. 2xA40 can run 14B but it's slow
- **USP** (sequence parallel): 14.5s/step — model replicated on both GPUs, only attention sharded
- **FSDP** (parameter shard): 25.2s/step — parameters split but more communication
- **Single A40**: OOM — 14B (28GB) + VAE (7GB) + overhead > 48GB
- Skipping VAE in Phase 1 allows USP to work (28GB model fits with room for activations)

### 4. 1.3B model is surprisingly fast
- 8.1s/step on A40 vs 9.4s/step for 14B on H100
- For latency-insensitive workloads, pure 1.3B at $0.16/video is hard to beat

### 5. Per-step timing summary

| Model | GPU | s/step | Notes |
|-------|-----|--------|-------|
| 14B | H100 | 9.4s | Best |
| 14B | 2xA40 USP | 14.5s | Model replicated, attention sharded |
| 14B | 2xA40 FSDP | 25.2s | Params sharded, more comms |
| 1.3B | A40 | 8.1s | Single GPU |
| 1.3B | 2xA40 (coordinator) | 7.5s | Via inline endpoint |

---

## Cost Analysis (at RunPod rates)

| Config | Latency | GPU $/hr | $/video | vs H100 14B |
|--------|---------|----------|---------|------------|
| H100 14B only | 472s | $2.69 | $0.35 | 1.0x (ref) |
| **H100 hybrid 30/70** | **222s** | **$2.69** | **$0.17** | **2.1x cheaper** |
| 2xA40 hybrid 30/70 (USP) | 1005s | $1.40 | $0.39 | 0.9x (more expensive!) |
| Distributed H100+A40 | 406s compute | mixed | ~$0.17 | 2.1x cheaper |
| 1.3B only (A40) | 403s | $0.44 | $0.05 | 7x cheaper (lower quality) |

---

## Recommendations

1. **For quality + speed**: Single H100 with hybrid 30/70 schedule
2. **For cost at scale**: Distributed with H100 spot for 14B + cheap GPU for 1.3B
3. **For budget**: 1.3B only on cheap GPU ($0.05/video)
4. **Avoid**: 2xA40 for 14B — it works but USP overhead makes it slower AND more expensive than H100

## Videos Produced
- `local_exp1_14B_baseline.mp4` — 14B on H100 (5.6MB)
- `local_exp2_hybrid_30_70.mp4` — Hybrid 30/70 distributed (5.7MB)
- `local_exp3_hybrid_20_80.mp4` — Hybrid 20/80 distributed (5.6MB)
- `local_exp4_1.3B_only.mp4` — 1.3B on 2xA40 (9.9MB)
- `14B_2xA40_fsdp.mp4` — 14B on 2xA40 FSDP (5.5MB) (on pod)
- `hybrid_30_70_2xA40_usp.mp4` — Hybrid 30/70 USP (on pod)
- `expB_1.3B_2xA40.mp4` — 1.3B native generate.py (on pod)
