# HXQ: A Calibration-Free Compression Substrate for Multi-Architecture Edge Inference

**Joshua P Fellows**
Echo Labs LLC | Michigan

**April 3, 2026**

---

## Abstract

Existing weight compression methods (GPTQ, AWQ, bitsandbytes) are designed and tested primarily on decoder-only Transformers. We present HXQ, a calibration-free vector quantization codec that compresses any `nn.Linear` layer through per-tensor k-means, optional 2D VQ with 12-bit index packing, and sparse sidecar correction --- requiring no architecture-specific calibration. We demonstrate near-lossless downstream performance across six architecture families, with the strongest evidence on Zamba2-7B hybrid SSM-Transformer (HellaSwag +0.27% vs dense), OLMoE 64-expert MoE (-0.16%), and CLIP ViT-L/14 vision-text (CIFAR-100 top-1 +0.27%). With the required Mamba CUDA fast path installed, HXQ buffered inference reaches 1,827 tok/s on RTX 4090; on RTX 3090, HXQ buffered inference reaches 646 tok/s versus 1,446 tok/s dense under the matched 50x512-token benchmark, at 60% VRAM reduction. We further demonstrate compressed multi-architecture co-resident inference: three models from different families sharing one materialization buffer across 683 modules at 10.3 GB on a single 24 GB GPU.

---

## 1. Introduction

Post-training quantization (PTQ) has become the standard method for reducing the memory footprint of large language models. Methods such as GPTQ [1], AWQ [2], and bitsandbytes [3] have made it routine to deploy 7B-parameter models on consumer GPUs by reducing weight precision from 16 bits to 4 bits per weight. However, the neural network architecture landscape is diversifying rapidly beyond the decoder-only Transformer. State space models (Mamba [4], Mamba-2 [5]), hybrid SSM-Transformer architectures (Zamba2 [6], Jamba [7]), Mixture-of-Experts models (Mixtral [8], OLMoE [9]), and multimodal vision-language models (CLIP [10], LLaVA) each introduce computational patterns that existing compression tools handle unevenly or not at all.

This architecture fragmentation creates a practical problem: a deployment engineer who needs to run a Zamba2 model alongside a CLIP encoder and a code-generation Transformer on a single edge GPU currently has no unified compression tool that handles all three. GPTQ and AWQ require calibration data and are tested primarily on Transformers. bitsandbytes supports any `nn.Linear` but uses naive round-to-nearest quantization, sacrificing quality. GGUF/llama.cpp covers the widest architecture range but requires per-architecture C++ engineering and is not pip-installable.

We introduce HXQ (Helix Quantization), a calibration-free weight compression codec that operates at the tensor level rather than the architecture level. HXQ treats each weight matrix as a standalone compression problem: run k-means on the weight values, store codebook indices, optionally apply 2D vector quantization to capture correlated weight pairs, pack indices into 12-bit representation, and correct outliers via sparse sidecar tensors. Because this pipeline requires no knowledge of the surrounding architecture --- no activation statistics, no Hessian information, no calibration dataset --- it applies identically to Transformer attention projections, Mamba state-space projections, MoE expert feedforward layers, CLIP cross-modal projections, and BERT self-attention weights.

Our contributions are:

1. **A unified compression codec spanning six architecture families**, demonstrated with 14 publicly available compressed models on HuggingFace, with paired dense-vs-compressed downstream evaluations on the strongest case studies (Zamba2-7B, OLMoE) and receipt-backed evaluations across the remaining families.

2. **2D vector quantization with 12-bit packed indices**, achieving 6 bits per weight with quality that exceeds bitsandbytes 4-bit NF4 on Zamba2-7B (PPL 5.015 vs 5.066) at 60% VRAM reduction.

3. **A shared weight materialization buffer** enabling compressed multi-architecture co-resident inference: three models from different architecture families loaded simultaneously on one GPU, sharing a single class-level buffer across all compressed modules.

4. **Supporting systems work on PolarQuant KV cache rotation** for hybrid SSM-Transformer compressed caches, achieving 93% MSE improvement on Zamba2 key tensors. In the public literature and toolchains surveyed here, we are not aware of prior PolarQuant evaluations on hybrid architectures.

Evidence depth varies by architecture family: the strongest paired downstream evaluations are reported for hybrid (Zamba2-7B) and MoE (OLMoE) case studies, with task-specific evaluations for vision and encoder-only families, and perplexity evaluations for the remaining models.

---

## 2. Related Work

### 2.1 Transformer-Centric Weight Compression

GPTQ [1] uses approximate second-order (Hessian) information for layer-wise weight quantization, achieving 3-4 bit compression with minimal perplexity loss. AWQ [2] identifies salient weight channels through activation profiling and protects them during quantization. Both require calibration data and are primarily tested on decoder-only Transformers, though AWQ has been applied to Mixtral (MoE) and some VLMs. bitsandbytes [3] provides the most architecture-agnostic Python library, replacing any `nn.Linear` with NF4 or int8 equivalents, but uses simple round-to-nearest quantization that is significantly less accurate than calibration-based methods at the same bit-width.

Advanced VQ-based methods have pushed quality at extreme compression ratios. QuIP# [11] uses E8 lattice codebooks with Hadamard incoherence processing. AQLM [12] uses multi-codebook additive quantization with joint block-wise optimization. QTIP [13] (NeurIPS 2024 Spotlight) achieves ultra-high-dimensional trellis-coded quantization. All are deeply coupled to Transformer block structures and have not been tested on SSMs, vision encoders, or hybrid architectures.

### 2.2 SSM-Specific Quantization

Recent work has established that state space models present unique quantization challenges. Quamba [14] (ICLR 2025) discovered highly sensitive feature maps within the selective scan mechanism and massive outlier activations absent in attention modules, with naive 8-bit quantization causing complete model collapse. MambaQuant [15] (ICLR 2025) demonstrated that standard rotation-based approaches (QuaRot) fail on Mamba due to inconsistent channel variances, requiring Karhunen-Loeve Transformation instead. Additional work includes Q-Mamba [16], QMamba [17], SSDi8, and Slender-Mamba [18]. Critically, we are not aware of public, maintained compressed checkpoint releases accompanying these methods in the way HXQ distributes artifacts through HuggingFace.

### 2.3 Vector Quantization for Neural Network Weights

VQ for weight compression dates to Gong et al. (2014) and was prominently featured in Deep Compression [19] (ICLR 2016 Best Paper). GPTVQ [20] (Qualcomm, ICML 2024) is the most directly relevant prior work: it explicitly explores 1D vs 2D VQ, demonstrating that pairing adjacent weights into 2-element vectors improves quantization quality because "vector quantization more closely fits 2D normal distributions." HXQ's 2D VQ shares this geometric insight but differs in three ways: (1) HXQ is calibration-free while GPTVQ uses Hessian-weighted optimization, (2) HXQ operates across six architecture families while GPTVQ tested only on Transformers, and (3) HXQ uses 12-bit packed indices (k=4096) while GPTVQ uses smaller codebooks.

### 2.4 KV Cache Compression

TurboQuant [21] (Google, ICLR 2026) represents the current state of the art for KV cache compression, achieving 6x reduction through PolarQuant rotation followed by scalar quantization. The PolarQuant rotation --- a random orthogonal transformation that spreads outlier-dimension energy uniformly across all coordinates --- is the core innovation. The secondary QJL component has been contested by six independent implementation teams who found that softmax exponentially amplifies QJL's variance, making MSE-only quantization superior for generation quality. KIVI [22], KVQuant [23], and GEAR provide complementary approaches. In the public literature and toolchains surveyed here, we are not aware of KV cache compression specifically evaluated on hybrid SSM-Transformer architectures, where only the attention layers (a minority of total layers) maintain KV caches while SSM layers maintain fixed-size recurrent state.

### 2.5 Multi-Model Inference

vLLM, HuggingFace TGI, and DeepSpeed-MII are the dominant LLM inference engines, and all serve one model per GPU instance. NVIDIA Triton supports multi-model co-location but treats each model as a black box with no cross-model memory optimization. HuggingFace Multi-Model Inference Endpoints load multiple models on one GPU but with independent memory spaces. Gemel [24] (NSDI 2023) demonstrated cross-model weight sharing for architecturally similar CNN models. In the public literature and toolchains surveyed here, we found no system that loads models from different architecture families (SSM, Transformer, vision encoder) simultaneously on one GPU with a shared decompression buffer.

---

## 3. The HXQ System

### 3.1 Codec Overview

HXQ compresses `nn.Linear` weight matrices through the following pipeline, applied independently to each tensor:

1. **Tensor classification**: A name-based policy classifies each tensor as compressible (attention, MLP, projection weights), exact (norms, embeddings, bias vectors, conv1d, SSM-specific tensors like A_log, D, dt_bias), or skip (position_ids, buffers).

2. **Per-tensor k-means**: For compressible tensors, k-means clustering (k=256 for scalar VQ, k=4096 for 2D VQ) partitions the weight values into codebook entries. No calibration data, activation statistics, or Hessian information is used --- clustering operates on weight values alone.

3. **Index assignment**: Each weight (or weight pair for 2D VQ) is assigned to its nearest codebook entry. The assignment is stored as integer indices.

4. **Sidecar outlier correction**: Weights with reconstruction error exceeding a threshold are stored as sparse (row, col, delta) triplets, providing exact correction for the most sensitive values.

5. **12-bit index packing**: For k=4096 codebooks (log2(4096)=12 bits per index), pairs of 12-bit indices are packed into 3 bytes, reducing index storage by 25% compared to uint16.

6. **Serialization**: Codebooks, packed indices, sidecar corrections, and metadata are stored in safetensors format compatible with HuggingFace model loading.

### 3.2 2D Vector Quantization

Standard scalar VQ assigns one codebook entry per weight. 2D VQ groups pairs of adjacent weights and clusters in R^2, capturing the joint distribution. With k=4096 and 12-bit index packing, each pair of weights costs 12 bits, yielding 6 bits per weight.

The advantage is measurable: on Zamba2-7B, 2D VQ k=4096 achieves +3.33% PPL versus +7.97% for scalar k=256 --- a 58% reduction in compression error. The benefit scales with model depth: mean per-tensor cosine similarity is 0.999756 at 7B (81 layers) versus 0.999632 at 1.2B (38 layers), indicating that per-tensor quality improves at scale while degradation is dominated by error compounding through depth.

### 3.3 Inference: Buffered Forward with Shared Materialization Buffer

The core inference mechanism is a buffered forward path. Weights remain compressed in VRAM (codebook + packed indices). During each forward pass, one layer at a time is decompressed into a shared buffer, a standard cuBLAS matmul executes at full tensor core speed, and the next layer overwrites the same buffer. This is architecturally identical to how bitsandbytes operates: compressed storage, per-forward dequantization, cuBLAS compute.

The key design decision is a **class-level shared buffer**: `HelixLinear._shared_buffer` is a single `torch.Tensor` allocated once and reused by ALL compressed modules across ALL models in the process. When multiple models are loaded (Section 5), they automatically share this buffer. The buffer is sized to the largest layer encountered and dynamically grows if a larger layer is loaded. VRAM cost is approximately 50-100 MB for a single buffer serving hundreds of modules.

### 3.4 Fused Triton Gather Kernel

The per-forward decompression cost is dominated by the index gather operation: reading packed uint8 bytes, extracting 12-bit indices, and looking up codebook entries. We implement a fused Triton kernel (`triton_gather_12bit.py`) that performs this in a single GPU kernel launch: packed bytes are read from global memory, 12-bit indices are extracted via bit manipulation, codebook entries (32 KB, fitting in shared memory) are looked up, and BF16 weights are written directly to the output buffer. No intermediate allocations occur.

Speed progression during development: 46 tok/s (naive tiled) to 281 tok/s (fused Triton matmul) to 570 tok/s (shared buffer, PyTorch gather) to **646 tok/s** (index_select optimization + shared buffer). With the Mamba CUDA fast path installed (`mamba-ssm` with `causal-conv1d`), end-to-end throughput reaches **1,827 tok/s on RTX 4090** in our verification harness; on RTX 3090, the matched 50x512-token benchmark reports 646 tok/s for HXQ buffered inference versus 1,446 tok/s dense.

Profiling reveals why: HXQ decompression (gather + sidecar + matmul) accounts for only **7.4% of forward-pass time** on Zamba2-7B. The remaining 92.6% is non-HelixLinear computation (Mamba scan, attention, layer normalization). Without the Mamba CUDA fast path, a naive Python sequential scan dominates the forward pass, producing only 202 tok/s regardless of decompression efficiency. The compressed representation reduces model VRAM from 14 GB to 5.7 GB, and the per-layer decompression overhead (0.93 ms per HelixLinear module vs 0.39 ms pure cuBLAS) is amortized across the full forward pass.

**Dependency note.** Zamba2 throughput numbers require `mamba-ssm>=2.2.2` and `causal-conv1d>=1.4.0` compiled against the local CUDA toolkit. Without these, HuggingFace falls back to a naive Python Mamba implementation that is 9x slower, regardless of compression method.

### 3.5 Architecture-Aware Tensor Policy

While the codec itself is architecture-agnostic, the tensor classification policy recognizes architecture-specific naming conventions to correctly identify which tensors should be compressed versus stored exactly. For SSM models, this includes Mamba-specific tensors (A_log, D, dt_bias, conv1d weights). For MoE models, expert gating weights and router projections are identified. For vision models, patch embeddings and position embeddings are stored exactly. This policy layer is the only component with architecture awareness; the compression and inference paths are identical regardless of architecture.

### 3.6 mamba-scan-lite: Memory-Efficient SSM Inference

A practical blocker for deploying hybrid SSM-Transformer models on consumer hardware is that HuggingFace's naive Mamba implementation materializes the entire SSM scan as a dense matrix, consuming gigabytes of memory and causing OOM on GPUs with less than 8 GB. We provide `mamba-scan-lite` (PyPI), a standalone package that monkey-patches HuggingFace's Zamba2 implementation with a sequential token-by-token scan using O(state_size) memory. This reduces SSM computation memory from gigabytes to megabytes, enabling Zamba2 models to run on 4 GB GPUs. The patch is architecture-correct (same mathematical computation) and requires only `import mamba_scan_lite` before model loading.

---

## 4. Experimental Results

All experiments use the same HXQ codec (helix-substrate v0.3.3, PyPI). Dense baselines and HXQ models are evaluated with identical settings on the same hardware.

**Note on evaluation regimes.** This section reports results from several distinct evaluation protocols that should not be compared directly across tables: (1) perplexity (PPL) via sliding-window log-likelihood on WikiText-2; (2) downstream task accuracy via lm-eval-harness multiple-choice scoring; (3) zero-shot image classification accuracy for CLIP; and (4) masked language modeling top-k accuracy for BERT. Each protocol measures different aspects of model quality. Compression ratios and VRAM measurements are consistent across all protocols.

### 4.1 Zamba2-7B-Instruct (Hybrid SSM-Transformer)

Zamba2-7B-Instruct [6] is a hybrid architecture with 32 Mamba-2 layers and 6 shared Transformer attention layers (81 total blocks). HXQ compresses 213 `nn.Linear` modules using 2D VQ k=4096 with 12-bit index packing (6 bits/weight).

**Perplexity** (WikiText-2, MAX_LENGTH=2048, STRIDE=512, RTX 3090):

| Configuration | PPL | Delta |
|---|---|---|
| Dense BF16 | 3.7218 | --- |
| HXQ 2D VQ k=4096 | 3.8457 | +3.33% |

**Downstream Tasks** (lm-eval-harness v0.4.11, acc_norm, RTX 3090):

| Task | Dense BF16 | HXQ 2D VQ | Delta |
|---|---|---|---|
| HellaSwag | 80.79% | 81.06% | +0.27% |
| ARC-Challenge | 59.39% | 58.11% | -1.28% |
| ARC-Easy | 83.21% | 81.90% | -1.31% |

**Throughput and Memory** (50 chunks x 512 tokens, RTX 3090, `mamba-ssm` fast path installed):

| Configuration | PPL | tok/s | VRAM Load | bits/wt |
|---|---|---|---|---|
| Dense BF16 | 4.821 | 1,446 | 14,032 MB | 16 |
| bnb 8-bit | 4.854 | 515 | 7,831 MB | 8 |
| bnb 4-bit NF4 | 5.066 | 1,579 | 5,129 MB | 4 |
| HXQ buffered (shared buffer) | 5.019 | 646 | 5,657 MB | 6 |
| HXQ fused (Triton matmul) | 5.023 | 284 | 6,080 MB | 8 |

An independent verification on RTX 4090 (same methodology, 50x512 tokens) measured **1,827 tok/s** for the HXQ buffered path with `mamba-ssm>=2.2.2` installed. The difference from the 3090 result (646 tok/s) reflects both the faster GPU and potential differences in Mamba CUDA kernel versions. Without `mamba-ssm` installed, the same 4090 benchmark measured only 202 tok/s due to HuggingFace's naive Python Mamba fallback consuming 92.6% of forward-pass time.

HXQ at 6 bits/weight achieves better quality than bnb 4-bit NF4 at comparable VRAM on the matched RTX 3090 benchmark (PPL 5.019 vs 5.066; VRAM 5,657 vs 5,129 MB). Separate RTX 4090 verification demonstrates that, with the required Mamba fast path installed, buffered HXQ inference can exceed 1.8k tok/s.

**Throughput methodology.** All tok/s measurements use the same harness: 50 non-overlapping 512-token chunks from WikiText-2, batch size 1, BF16 compute dtype. Throughput is measured as total tokens processed divided by total wall-clock time including all decompression overhead. The PPL values in this table differ from the strided-window PPL above because they use non-overlapping windows (less context per token) rather than overlapping 2048-token windows with 512-token stride. Both dense and HXQ configurations use identical chunking, so relative comparisons within this table are valid. Profiling on RTX 4090 shows HXQ decompression (gather + sidecar + cuBLAS matmul) accounts for 7.4% of total forward-pass time; the remaining 92.6% is non-compressed computation (Mamba scan, attention, normalization).

### 4.2 OLMoE-1B-7B-Instruct (Mixture of Experts)

OLMoE [9] has 64 experts with top-8 gating (6.9B total parameters, 1.3B active). HXQ compresses 3,152 modules: 3,072 expert layers + 64 attention + 16 router gates.

**Downstream Tasks** (lm-eval-harness v0.4.11, acc_norm, RTX 3090):

| Task | Dense BF16 | HXQ | Delta |
|---|---|---|---|
| HellaSwag | 78.92% | 78.76% | -0.16% |
| ARC-Challenge | 52.13% | 52.05% | -0.08% |
| ARC-Easy | 75.72% | 76.85% | +1.14% |

All deltas are within standard error (~0.4-1.5%). VRAM: Dense 13,886 MB, HXQ 7,540 MB (1.84x reduction). We are not aware of another post-training compressed OLMoE checkpoint publicly available on HuggingFace at the time of writing.

### 4.3 CLIP ViT-L/14 (Vision-Text Encoder)

CLIP [10] is a dual-encoder model with 427M parameters across 218 compressed linear layers (text encoder + vision encoder). HXQ uses scalar VQ k=256 (8 bits/weight).

**Zero-Shot Classification** (CIFAR-100, 10,000 images, RTX 3090):

| Metric | Dense | HXQ | Delta |
|---|---|---|---|
| Top-1 Accuracy | 72.48% | 72.75% | +0.27% |
| Top-5 Accuracy | 91.41% | 91.64% | +0.23% |

Storage: 1.6 GB dense to 447 MB HXQ (3.6x compression). The compressed model slightly outperforms dense, likely due to regularization effects of quantization.

### 4.4 BERT-base-uncased (Encoder-Only Transformer)

BERT [25] is a 12-layer encoder-only masked language model with 110M parameters. HXQ compresses 75 linear layers.

**Masked Language Modeling** (WikiText-2, 500 masked predictions):

| Metric | Dense | HXQ | Delta |
|---|---|---|---|
| MLM Top-1 | 61.40% | 61.00% | -0.40% |
| MLM Top-5 | 77.60% | 77.00% | -0.60% |

Storage: 421 MB dense to 106 MB HXQ (4.0x compression).

### 4.5 Cross-Architecture Summary

| Family | Model | Compression | Best Eval Delta |
|---|---|---|---|
| Decoder Transformer | Qwen 1.5B-14B, TinyLlama | 1.5-4.0x | PPL within noise |
| Pure SSM | Mamba-130M, Mamba2-1.3B | 2.1-3.8x | PPL receipted |
| Hybrid SSM+Transformer | Zamba2 1.2B, 2.7B, 7B | 1.7-1.9x | HellaSwag +0.27% |
| Mixture of Experts | OLMoE-1B-7B | 1.9x | HellaSwag -0.16% |
| Vision-Text | CLIP ViT-L/14 | 3.6x | Top-1 +0.27% |
| Encoder-Only | BERT-base | 4.0x | MLM -0.40% |

14 models across 6 architecture families, all compressed with the same codec (`pip install helix-substrate`). Auxiliary packages provide architecture-specific fast paths: `mamba-scan-lite` for memory-efficient SSM inference on small GPUs, and `mamba-ssm`/`causal-conv1d` for CUDA-accelerated Mamba inference on Ampere+ hardware.

> **Note on Table 4.** This summary spans different evaluation protocols by architecture family. "Best Eval Delta" reports the most favorable metric from each family's evaluation suite. Evidence depth varies: Zamba2-7B and OLMoE have full paired downstream evaluations; CLIP and BERT have task-specific evaluations; pure SSM and decoder Transformer families have perplexity evaluations. Exact protocols are detailed per-family in Sections 4.1--4.4 and in the JSON receipts accompanying each HuggingFace model.

### 4.6 Divergence Probe: SSM Error Dampening

We measured layer-wise quantization error amplification by comparing dense and HXQ-compressed model outputs at each layer boundary. For each layer *l*, we computed the ratio of output divergence (L2 norm of the difference between dense and compressed hidden states) to input divergence at that layer. Total amplification is the product of per-layer ratios across the full model depth. This was measured on 100 randomly sampled 512-token sequences from WikiText-2:

- **TinyLlama (Transformer, 22 layers)**: 33.4x total error amplification
- **Mamba-130M (SSM, 24 layers)**: 8.7x total amplification; layers 8-17 actively attenuate error

SSM architectures dampen quantization error approximately 4x better than Transformers. This finding is consistent with the recurrent state dynamics acting as a low-pass filter on quantization noise, and explains why calibration-free compression achieves competitive results on SSM and hybrid architectures specifically.

We note that this finding appears to contradict Quamba [14] and MambaQuant [15], which report that SSMs are harder to quantize. The distinction is that those works target activation quantization (W8A8), where SSM-specific outlier patterns cause collapse, while HXQ performs weight-only quantization where the recurrent dynamics smooth weight perturbations during forward propagation.

---

## 5. Multi-Model Compressed Co-Resident Inference

### 5.1 System Architecture

We demonstrate three HXQ-compressed models loaded simultaneously into a single process on one RTX 3090 (24 GB):

| Lobe | Model | Architecture | HelixLinear Modules | VRAM |
|---|---|---|---|---|
| Language | Zamba2-7B-Instruct-HXQ | Hybrid SSM+Transformer | 213 | 5,428 MB |
| Code | Qwen2.5-Coder-3B-HXQ | Decoder Transformer | 252 | 3,475 MB |
| Vision | CLIP-ViT-L/14-HXQ | Dual Encoder | 218 | 527 MB |
| **Total** | | **3 families** | **683** | **9,242 MB** |

All 683 HelixLinear modules share a single class-level materialization buffer (`HelixLinear._shared_buffer`), confirmed via Python object identity. Total VRAM during inference reaches 10.3 GB (the difference from the 9,242 MB model loading total reflects runtime activations, KV cache allocations, and server process overhead), leaving 14.2 GB headroom on a 24 GB GPU.

### 5.2 Routing and Integration

An input router dispatches queries to the appropriate model based on content analysis: code-related keywords route to the Qwen code lobe, image inputs route to CLIP, and all other queries route to Zamba2. The system integrates:

- **Soulfile identity injection**: A JSON configuration defining the system's name ("Echo") and behavioral principles, prepended to every language generation prompt.
- **EchoMemory**: SQLite-based interaction storage with keyword search, enabling cross-session context retrieval.
- **FGIP knowledge graph**: 1,896 nodes and 3,411 edges of structured domain knowledge (supply chain analysis, investment thesis nodes), searchable via FTS5 and automatically injected into prompts when relevant keywords are detected.

### 5.3 Smoke Test Results

A scripted 15-query test battery covering all modalities, with pass/fail determined by automated string matching for factual queries, syntax validation for code generation, and label correctness for vision classification:

| Category | Tests | Result |
|---|---|---|
| Language reasoning | 3 queries | Correct, 4.9 tok/s |
| Code generation | 2 queries (auto + explicit routing) | Clean Python, 7.6 tok/s |
| Vision classification | 1 image | Correct label, 195ms |
| Memory store/recall | 3 queries | Store and retrieve confirmed |
| FGIP knowledge retrieval | 2 queries | Relevant nodes returned |
| Identity/soulfile | 2 queries | Correct self-identification |
| Multi-turn conversation | 1 query | Context maintained |
| Embeddings | 1 query | 768-dim vector returned |
| **Total** | **15/15 pass** | |

---

## 6. PolarQuant KV Cache Compression for Hybrid Architectures

### 6.1 Motivation

Zamba2's hybrid architecture has 38 layers but only 6 attention layers that maintain KV caches. The remaining 32 Mamba layers maintain fixed-size recurrent state (~2 MB total). KV cache compression thus targets only the attention minority, but these layers' caches grow linearly with sequence length and become the VRAM bottleneck at long contexts.

### 6.2 Integration

We integrate PolarQuant rotation [21] into our online KV cache compression pipeline. Before scalar VQ quantization, each KV vector is multiplied by a random orthogonal matrix (generated per-layer via QR decomposition with a deterministic seed). This spreads outlier-dimension energy uniformly, improving centroid utilization. On decompression, the inverse rotation (Q^T, since Q is orthogonal) restores the original coordinate system.

### 6.3 Results on Zamba2

Benchmarked on real KV cache tensors captured from Zamba2-1.2B inference across three diverse prompts:

| Metric | Without PolarQuant | With PolarQuant | Improvement |
|---|---|---|---|
| K MSE (avg across layers) | baseline | -93% | 93% reduction |
| Attention top-16 agreement | 0.72 | 0.97 | +35% |
| VRAM overhead | 0 | 350 KB (rotation matrices) | Negligible |

Layer-by-layer analysis shows PolarQuant is most effective at deeper layers (5-35) where channel outlier concentration is highest. Layer 0-1 shows minimal improvement (<10%) because early layers have relatively uniform channel distributions. PolarQuant is set as the default in the released configuration.

In the public literature and toolchains surveyed here, we are not aware of prior evaluations of PolarQuant rotation on hybrid SSM-Transformer KV caches.

---

## 7. Limitations

We acknowledge several limitations that constrain the scope of our claims:

**Throughput depends critically on non-HXQ dependencies.** On Zamba2-7B, HXQ decompression accounts for only 7.4% of forward-pass time. The remaining 92.6% is Mamba scan, attention, and normalization. Without `mamba-ssm` CUDA kernels installed, throughput drops from 1,827 tok/s to 202 tok/s due to HuggingFace's naive Python fallback --- a 9x penalty entirely outside the compression pipeline. On devices without Triton or CUDA Mamba kernels (e.g., Turing-generation GPUs), the fallback path achieves only 2-3 tok/s for Zamba2 models. The mamba-scan-lite sequential scan trades speed for memory compatibility on 4 GB devices.

**Not all architecture families are equally benchmarked.** Zamba2-7B and OLMoE have full downstream evaluation suites (HellaSwag, ARC-Challenge, ARC-Easy with paired dense comparisons). CLIP has zero-shot classification. BERT has masked LM accuracy. The pure SSM models (Mamba-130M, Mamba2-1.3B) have only PPL evaluations. Depth of evidence varies across the portfolio.

**No matched GPTQ/AWQ comparisons on hybrid or MoE models.** No GPTQ or AWQ compressed Zamba2 or OLMoE checkpoints exist on HuggingFace for direct comparison. Our comparison to bitsandbytes is the only apples-to-apples baseline available for these architectures. For Transformer models where GPTQ/AWQ exist, HXQ at 6-8 bits/weight occupies a different quality-compression tradeoff than 3-4 bit methods.

**Calibration-free compression trades peak quality for universality.** Methods using Hessian information (GPTQ) or activation statistics (AWQ) achieve better quality at the same bit-width on Transformer models. HXQ's advantage is not per-architecture quality leadership but cross-architecture coverage from a single tool.

**CNN support is codec-level only.** ResNet-50 weights compress successfully at the tensor level (54 tensors, cosine >0.999, 3.9x compression), but the HuggingFace inference path requires a `HelixConv2d` module that does not yet exist. Vision Transformer and fully-connected models are fully supported.

**The multi-model system is a demonstration, not a production serving system.** It lacks request batching, concurrent request handling, dynamic model loading/unloading, and the optimizations present in vLLM or Triton. Its purpose is to establish feasibility of compressed multi-architecture co-residency, not to compete with production inference servers.

---

## 8. Conclusion

HXQ demonstrates that a single calibration-free compression codec can operate across the major neural network architecture families deployed today. The compression primitive operates on weight tensors without explicit architecture-specific calibration, although empirical tolerance to compression still varies by architecture family and tensor type. Architecture-specific concerns (SSM state tensors, MoE gating, vision patch embeddings) are handled by a lightweight tensor classification policy, not by changes to the codec itself.

The practical implications are:

1. **Unified edge deployment**: A single `pip install` provides compression and inference for Transformers, SSMs, hybrids, MoE, vision, and encoder-only models. This eliminates the need for per-architecture compression toolchains.

2. **Multi-model co-residency**: The shared materialization buffer design enables multiple compressed models from different architecture families to coexist on a single consumer GPU with minimal VRAM overhead beyond compressed storage.

3. **SSM-favorable compression characteristics**: The divergence probe finding that SSMs dampen quantization error 4x better than Transformers suggests that the shift toward SSM and hybrid architectures may make post-training compression more effective, not less --- provided weight-only (not activation) quantization is used.

All models, code, benchmark receipts, and the inference server are publicly available. Models are hosted at HuggingFace under the EchoLabs33 organization. The compression library is available via `pip install helix-substrate` (v0.3.3). The memory-efficient Mamba scan is available via `pip install mamba-scan-lite` (v0.1.0).

---

## 9. Reproducibility

All results can be independently verified from publicly available artifacts:

| Component | Location | Version |
|---|---|---|
| Compression & inference library | `pip install helix-substrate` | v0.3.3 |
| Memory-efficient Mamba scan | `pip install mamba-scan-lite` | v0.1.0 |
| Compressed model checkpoints | HuggingFace: EchoLabs33 | 14 models |
| Source code | GitHub: echo313unfolding/helix-substrate | commit-dated |
| Benchmark receipts | JSON files in each HF model repository | per-model |
| Hardware | NVIDIA RTX 3090 24 GB (bench_6config), RTX 4090 24 GB (speed verification), Quadro T2000 4 GB (edge) | — |

Each benchmark receipt is a machine-readable JSON file containing: hardware specifications, PyTorch and transformers versions, exact hyperparameters (batch size, sequence length, stride, number of chunks), SHA256 hashes of input data, wall-clock and CPU time, and VRAM measurements. Receipt filenames include ISO timestamps. Any result in this paper can be traced to a specific receipt and reproduced on equivalent hardware. The whitepaper source (markdown) and compiled PDF are maintained at `echo313unfolding/hxq-whitepaper` on GitHub.

---

## References

[1] Frantar, E., et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-Trained Transformers." ICLR 2023.

[2] Lin, J., et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys 2024.

[3] Dettmers, T., et al. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." NeurIPS 2022; "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.

[4] Gu, A. and Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces." arXiv 2312.00752, 2023.

[5] Dao, T. and Gu, A. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality." ICML 2024.

[6] Glorioso, P., et al. "The Zamba2 Suite: Technical Report." arXiv 2411.15242, 2024.

[7] Lieber, O., et al. "Jamba: A Hybrid Transformer-Mamba Language Model." ICLR 2025.

[8] Jiang, A., et al. "Mixtral of Experts." arXiv 2401.04088, 2024.

[9] Muennighoff, N., et al. "OLMoE: Open Mixture-of-Experts Language Models." arXiv 2409.02060, 2024.

[10] Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.

[11] Tseng, A., et al. "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks." ICML 2024.

[12] Egiazarian, V., et al. "Extreme Compression of Large Language Models via Additive Quantization." ICML 2024.

[13] Tseng, A., et al. "QTIP: Quantization with Trellises and Incoherence Processing." NeurIPS 2024 Spotlight.

[14] Chiang, H.-Y., et al. "Quamba: A Post-Training Quantization Recipe for Selective State Space Models." ICLR 2025.

[15] Xu, Z., et al. "MambaQuant: Quantizing the Mamba Family with Variance Aligned Rotation Methods." ICLR 2025.

[16] Liu, Y., et al. "Q-Mamba: Towards more efficient Mamba models via Post-Training Quantization." ACL 2025 Findings.

[17] Zhang, Y., et al. "QMamba: Post-Training Quantization for Vision State Space Models." arXiv 2501.13624, 2025.

[18] Wu, S., et al. "Slender-Mamba: Fully Quantized Mamba in 1.58 Bits From Head to Toe." COLING 2025.

[19] Han, S., Mao, H., and Dally, W. "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding." ICLR 2016 Best Paper.

[20] Van Baalen, M., et al. "GPTVQ: The Blessing of Dimensionality for LLM Quantization." ICML 2024.

[21] Zandieh, A., et al. "TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate." ICLR 2026.

[22] Liu, Z., et al. "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache." ICML 2024.

[23] Hooper, C., et al. "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization." NeurIPS 2024.

[24] Padmanabhan, A., et al. "Gemel: Model Merging for Memory-Efficient, Real-Time Video Analytics at the Edge." NSDI 2023.

[25] Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL 2019.

---

## Appendix A: Biological Coding Metaphor

We note a structural parallel between VQ weight compression and biological genetic coding that, to our knowledge, has not been drawn in the literature. DNA uses a 4-letter alphabet (A, T, G, C) producing 64 codons that map to 20 amino acids via a biological lookup table (the genetic code). VQ weight compression maps continuous weight values to discrete codebook indices that reference learned codebook entries. The codebook is analogous to the genetic code; the indices are analogous to the DNA sequence; the reconstructed weight matrix is analogous to the expressed protein.

This analogy extends to practical operations: swapping a codebook on the same index structure (analogous to altering the genetic code while preserving DNA sequence) would produce different model behavior without reloading the index tensor --- a 218 KB codebook swap versus a 1.5 GB weight reload. While we have not experimentally validated codebook-space interpolation, the HXQ infrastructure makes this mechanically feasible and it remains an area of future investigation.

The name "Helix" in helix-substrate derives from this biological inspiration, and the codec was independently conceived from DNA compression metaphors before the authors encountered the VQ literature. This independent derivation path is documented in LINEAGE.md (335 lines) in the project repository.

---

## Appendix B: Artifact Availability

| Artifact | Location |
|---|---|
| Compression library | `pip install helix-substrate` (v0.3.3) |
| Mamba scan patch | `pip install mamba-scan-lite` (v0.1.0) |
| Compressed models | HuggingFace: EchoLabs33 (14 models) |
| Source code | GitHub: echo313unfolding/helix-substrate |
| Benchmark receipts | JSON files in each HF model repository |
| This document | GitHub: echo313unfolding/hxq-whitepaper |

All benchmark receipts are machine-readable JSON files containing hardware specifications, software versions, exact hyperparameters, and SHA256 hashes of input data. Any result reported in this paper can be independently reproduced from the corresponding receipt.

---

## Appendix C: Scope of Claims

To prevent misinterpretation, we explicitly state what this paper does **not** claim:

- **We do not claim state-of-the-art compression quality on any single architecture.** Calibration-based methods (GPTQ, AWQ) and advanced VQ methods (QuIP#, AQLM, QTIP) achieve better quality-per-bit on Transformer models. HXQ's contribution is cross-architecture coverage, not per-architecture quality leadership.

- **We do not claim that no proprietary or internal system exists with similar capabilities.** Large technology companies may have internal multi-model compressed inference systems. Our claim is limited to publicly available, publicly documented, and publicly installable tools and artifacts as surveyed in this paper.

- **We do not claim that every architecture family has equally deep evaluation.** Zamba2-7B and OLMoE have full downstream task suites. Other families have less comprehensive benchmarks. The depth of evidence varies and is documented per-family.

- **We do not claim that throughput results transfer across dependency configurations.** The 1,827 tok/s result requires `mamba-ssm>=2.2.2` with CUDA fast path kernels. Without this dependency, the same model on the same GPU produces 202 tok/s. Throughput comparisons between HXQ and dense are only valid when both use identical Mamba kernel configurations. The 3090 bench_6config table (646 tok/s HXQ vs 1,446 dense) and the 4090 verification (1,827 tok/s) were conducted under different hardware and potentially different Mamba kernel versions.

- **We do not claim that "unified" means "every tensor treated identically."** HXQ uses a name-based tensor policy to classify tensors as compressible, exact, or skip. The codec pipeline is unified; the tensor classification is architecture-aware. This is analogous to how bitsandbytes replaces `nn.Linear` universally but does not modify `nn.Embedding` or `nn.LayerNorm`.
