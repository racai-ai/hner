# SMM4H NER — `hner`

A toolkit for **Named Entity Recognition (NER)** in social-media health text, built for the [Social Media Mining for Health (SMM4H)](https://healthlanguageprocessing.org/smm4h/) shared tasks.

`hner` supports multiple model architectures — from BERT adapters and fine-tuned transformers to GLiNER, variational models, semi-supervised FixMatch, a BiLSTM baseline, and an LLM-based zero-shot/few-shot approach — all behind a single command-line interface.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Training a model](#training-a-model)
  - [Running inference](#running-inference)
  - [Evaluating predictions](#evaluating-predictions)
- [Data Format](#data-format)
- [Supported Model Types](#supported-model-types)
- [CLI Reference](#cli-reference)
- [License](#license)
- [Contributing](#contributing)

---

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0 (GPU strongly recommended for training)
- For the `llm_ner` model type: a running [Ollama](https://ollama.com/) server

---

## Installation

```bash
# Clone the repository
git clone https://github.com/racai-ai/hner.git
cd hner

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

---

## Quick Start

### Training a model

Train a BERT adapter model on your dataset:

```bash
hner train data/train.csv data/dev.csv models/my_model --type bert_adapter
```

Train with a custom loss blend (CrossEntropy + Dice) for 30 epochs:

```bash
hner train data/train.csv data/dev.csv models/my_model \
    --type bert_adapter \
    --alpha-ce 0.7 --alpha-dice 0.3 \
    --max-training-epochs 30
```

Optimize an LLM prompt with GEPA (requires Ollama):

```bash
hner train data/train.csv data/dev.csv models/llm_prompt \
    --type llm_ner \
    --ollama-model gemma3:27b \
    --max-metric-calls 30
```

### Running inference

Run inference with a trained model:

```bash
hner test data/test.csv predictions.csv --type bert_adapter --model models/my_model
```

Zero-shot inference with an LLM (no `--model` needed):

```bash
hner test data/test.csv predictions.csv --type llm_ner --ollama-model gemma3:27b
```

### Evaluating predictions

Compute strict and relaxed SMM4H NER metrics on a CSV file that contains both gold and predicted BIO tag columns:

```bash
hner evaluate predictions.csv --gold-col ner_tags_str --pred-col prediction
```

---

## Data Format

Input CSV files must contain at least the following columns:

| Column | Description |
|--------|-------------|
| `ID` | Unique sentence identifier |
| `tokens` | Python-list string of tokens, e.g. `"['I', 'took', 'ibuprofen']"` |
| `ner_tags` | Python-list string of integer BIO tag IDs (for training/evaluation) |
| `ner_tags_str` | Python-list string of BIO tag strings, e.g. `"['O', 'B-Drug', 'O']"` (for evaluation) |

---

## Supported Model Types

| `--type` | Description |
|----------|-------------|
| `bert_adapter` | BERT with adapter layers (parameter-efficient fine-tuning) |
| `bert_finetuned` | Full BERT fine-tuning |
| `gliner` | [GLiNER](https://github.com/urchade/GLiNER) generalised NER model |
| `bert_variational` | Variational BERT with uncertainty estimation |
| `bert_fixmatch` | Semi-supervised FixMatch with unlabeled data |
| `lstm` | BiLSTM baseline with optional word2vec embeddings |
| `llm_ner` | LLM zero-shot/few-shot via Ollama + optional GEPA prompt optimisation |

---

## CLI Reference

```
hner <command> [options]

Commands:
  train      Train a model and save it to disk
  test       Run inference and write predictions to a CSV file
  evaluate   Compute strict + relaxed NER metrics
  augment    Augment a dataset using an LLM via Ollama
  evolve     Run GEPA + DSPy data-augmentation evolution
  analyze    Analyse train/dev datasets and write an entity report
  compare    Compare two annotated CSV files and report divergences
```

Run `hner <command> --help` for full details on each sub-command.

---

## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) guide before opening a pull request.
