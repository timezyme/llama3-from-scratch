# Milestone 2-3 Test Fixtures

Binary golden files for verifying Part 2 operators against a trusted reference.

## Convention

Each fixture records:
- **source**: prompt text or synthetic tensor definition
- **layer**: layer number (if applicable)
- **tensor**: name and shape
- **dtype**: always FP32 after conversion
- **tolerance**: comparison epsilon (default 1e-2)
- **reference**: how the golden output was produced

## Fixture files

(To be populated in Phases 1-4.)
