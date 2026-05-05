# Todo

## Plan
Start with foundational modules that other modules depend on (logging, data types), then build up to core functionality (math utils, verifiers), then utilities (RL utils, general utils), and finally dataset transformation which depends on multiple prior modules.

## Tasks
- [ ] Task 1: Implement logging configuration that provides consistent formatted logging across the project with support for distributed training ranks (logger_utils + tests)
- [ ] Task 2: Implement data container classes for tokenized data, generation results, request info, and collated batch data with device transfer capabilities (data_types + tests)
- [ ] Task 3: Implement math utilities for extracting and normalizing LaTeX math expressions including boxed answer extraction, string normalization, and equivalence checking (math_utils + tests)
- [ ] Task 4: Implement ground truth verifiers for evaluating model outputs including GSM8K number extraction, math answer verification, and F1 score calculation (ground_truth_utils + tests)
- [ ] Task 5: Implement RL utilities including a Timer for profiling operations, sequence packing for efficient batch processing, and advantage calculation using GAE (rl_utils + tests)
- [ ] Task 6: Implement core utilities including helper functions like repeat_each, metrics tracking, checkpoint management, GPU specs, and disk usage warnings (utils + tests)
- [ ] Task 7: Implement launch utilities for subprocess execution, cloud storage operations, and workspace validation (launch_utils + tests)
- [ ] Task 8: Implement dataset transformation pipelines including tokenization configurations, transform functions for SFT/DPO/RLVR data, and label masking (dataset_transformation + tests)
