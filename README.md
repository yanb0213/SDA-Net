# SDA-Net: A Chinese Character Stroke Extraction Network Based on Dual-Layer Attention Mechanism

> Official implementation of SDA‑Net – a novel stroke extraction network for Chinese characters.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Introduction

SDA‑Net is a Chinese character stroke extraction network based on a dual‑layer attention mechanism. It addresses two core challenges in stroke extraction: **ambiguous pixel attribution in crossing regions** and **stroke fractures in non‑crossing regions**.

The network employs **three independent encoders** to extract features from crossing regions, non‑crossing regions, and the original image respectively, suppressing inter‑region interference from the very beginning. A **dual‑layer attention module** is then introduced:

- **Spatial Cross‑Region Attention (SCRA)** – establishes semantic correlations between crossing and non‑crossing regions to resolve attribution ambiguity at intersections.
- **Spatial Single‑Region Attention (SSRA)** – enhances intra‑region feature consistency to prevent stroke fractures in non‑crossing areas.

Extensive experiments on multiple public datasets demonstrate that SDA‑Net significantly outperforms state‑of‑the‑art methods.

📄 **Paper status**: Submitted. The code will be released upon acceptance.

---

## 🔧 Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- CUDA 11.3 (GPU recommended)

Install dependencies:
```bash
pip install -r requirements.txt
