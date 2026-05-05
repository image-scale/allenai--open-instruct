# Goal

## Project
open-instruct — a python project.

## Description
Open Instruct is a library for instruction-tuning and post-training large language models. It provides tools for:

1. Dataset handling - Loading, mixing, and transforming instruction datasets for training
2. Math evaluation utilities - Parsing and normalizing LaTeX math expressions for answer verification
3. Ground truth verification - Evaluating model outputs against ground truth labels (GSM8K, Math, F1 scoring)
4. Reinforcement learning utilities - Sequence packing, advantage calculation, and timing utilities
5. Core training utilities - Metrics tracking, checkpoint management, GPU specs, and general helpers
6. Dataset transformation - Tokenization pipelines and data processing for SFT, DPO, and RLVR training

## Scope
- 8 production source files to implement
- 8 test files to write
- Focus on core CPU-testable functionality
