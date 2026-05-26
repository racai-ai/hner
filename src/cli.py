"""CLI entry point for the SMM4H NER system.

Usage
-----
Train::

    hner train <input_file> <validation_file> <output_model> --type <model_type>
        [--alpha-ce 0.5] [--alpha-dice 0.5] [--max-training-epochs 50]

    # For the llm_ner model type, additional flags are available:
    hner train <input_file> <validation_file> <output_model> --type llm_ner
        [--ollama-model gemma3:27b] [--ollama-base-url http://localhost:11434]
        [--max-metric-calls 30] [--eval-sample-size 50]

Test (inference)::

    hner test <input_file> <output_file> --type <model_type> [--model <model_file>]

    # For the llm_ner model type in zero-shot mode:
    hner test <input_file> <output_file> --type llm_ner
        [--ollama-model gemma3:27b] [--ollama-base-url http://localhost:11434]

Augment::

    hner augment <input_file> <output_file> [--strategy negate] [--model mistral]

Evolve (GEPA + DSPy data augmentation)::

    hner evolve <train_csv> <dev_csv> [--ollama-model gemma3:27b] [--max-metric-calls 100]

Evaluate (official SMM4H strict + relaxed metrics)::

    hner evaluate <input_file> --gold-col <col> --pred-col <col>

Analyze (dataset entity analysis)::

    hner analyze <trainset_file> <devset_file> <report_file> [--fuzzy-threshold 0.5]

Supported model types: bert_adapter, bert_finetuned, gliner, bert_variational, bert_fixmatch, lstm, llm_ner
"""

import argparse
import sys

from data.augmenter import STRATEGY_REGISTRY, DEFAULT_MODEL, augment
from data.analyzer import analyze
from data.comparator import compare
from training.trainer import MODEL_REGISTRY, Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hner",
        description="SMM4H NER — train and test NER models for the shared task.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # ------------------------------------------------------------------ train
    train_parser = subparsers.add_parser(
        "train",
        help="Train a model and save it to disk.",
    )
    train_parser.add_argument("input_file", help="Path to the training CSV file.")
    train_parser.add_argument("validation_file", help="Path to the validation CSV file.")
    train_parser.add_argument("output_model", help="Directory where the trained model will be saved.")
    train_parser.add_argument(
        "--type",
        dest="model_type",
        required=True,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model architecture to use.",
    )
    train_parser.add_argument(
        "--metric",
        dest="metric",
        default="f1",
        choices=["f1", "relaxed_f1", "avg_f1"],
        help=(
            "Metric used to select the best model checkpoint during training. "
            "Use 'f1' for strict seqeval F1 (default), 'relaxed_f1' for "
            "overlap-based relaxed F1, or 'avg_f1' for the average of both."
        ),
    )
    train_parser.add_argument(
        "--unlabeled-file",
        dest="unlabeled_file",
        default=None,
        help="Path to the unlabeled JSONL data (required for FixMatch).",
    )
    train_parser.add_argument(
        "--word2vec-file",
        dest="word2vec_file",
        default=None,
        help=(
            "Path to a word2vec text-format file used to initialise the "
            "frozen E_w2v embeddings in the LSTM model.  Each line after "
            "the header should contain a word followed by its 300-dimensional "
            "vector values."
        ),
    )
    train_parser.add_argument(
        "--alpha-ce",
        dest="alpha_ce",
        type=float,
        default=0.5,
        help=(
            "Weight for the CrossEntropy loss component (default: 0.5). "
            "A value of 0 disables CrossEntropy loss."
        ),
    )
    train_parser.add_argument(
        "--alpha-dice",
        dest="alpha_dice",
        type=float,
        default=0.5,
        help=(
            "Weight for the Dice loss component (default: 0.5). "
            "A value of 0 disables Dice loss."
        ),
    )
    train_parser.add_argument(
        "--max-training-epochs",
        dest="max_training_epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50).",
    )
    train_parser.add_argument(
        "--ollama-model",
        dest="ollama_model",
        default=None,
        help=(
            "Ollama model identifier for the llm_ner model type "
            "(e.g. 'gemma3:27b').  Ignored for other model types."
        ),
    )
    train_parser.add_argument(
        "--ollama-base-url",
        dest="ollama_base_url",
        default=None,
        help=(
            "Ollama server base URL for the llm_ner model type "
            "(default: http://localhost:11434).  Ignored for other model types."
        ),
    )
    train_parser.add_argument(
        "--max-metric-calls",
        dest="max_metric_calls",
        type=int,
        default=None,
        help=(
            "GEPA budget: total evaluate() calls for llm_ner prompt "
            "optimisation (default: 30).  Ignored for other model types."
        ),
    )
    train_parser.add_argument(
        "--eval-sample-size",
        dest="eval_sample_size",
        type=int,
        default=None,
        help=(
            "Number of validation sentences sampled per GEPA evaluation "
            "for the llm_ner model type (default: 50).  "
            "Ignored for other model types."
        ),
    )

    # ------------------------------------------------------------------ test
    test_parser = subparsers.add_parser(
        "test",
        help="Run inference with a trained model and write predictions.",
    )
    test_parser.add_argument("input_file", help="Path to the test CSV file.")
    test_parser.add_argument("output_file", help="Path where the output CSV will be written.")
    test_parser.add_argument(
        "--type",
        dest="model_type",
        required=True,
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model architecture to use.",
    )
    test_parser.add_argument(
        "--model",
        dest="model_file",
        default=None,
        help="Path to a trained model directory. If omitted, the default zero-shot model is used.",
    )
    test_parser.add_argument(
        "--ollama-model",
        dest="ollama_model",
        default=None,
        help=(
            "Ollama model identifier for zero-shot llm_ner inference "
            "(e.g. 'gemma3:27b').  Ignored for other model types."
        ),
    )
    test_parser.add_argument(
        "--ollama-base-url",
        dest="ollama_base_url",
        default=None,
        help=(
            "Ollama server base URL for zero-shot llm_ner inference "
            "(default: http://localhost:11434).  Ignored for other model types."
        ),
    )

    # --------------------------------------------------------------- augment
    augment_parser = subparsers.add_parser(
        "augment",
        help="Augment a dataset using an LLM via Ollama and write results to a new CSV.",
    )
    augment_parser.add_argument("input_file", help="Path to the source CSV file.")
    augment_parser.add_argument("output_file", help="Path where the augmented CSV will be written.")
    augment_parser.add_argument(
        "--strategy",
        dest="strategy",
        default="negate",
        choices=sorted(STRATEGY_REGISTRY.keys()),
        help="Augmentation strategy to apply (default: negate).",
    )
    augment_parser.add_argument(
        "--model",
        dest="model",
        default=DEFAULT_MODEL,
        help=f"Ollama model name to use (default: {DEFAULT_MODEL}).",
    )

    # -------------------------------------------------------------- compare
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two annotated CSV files and write diverging examples to a text file.",
    )
    compare_parser.add_argument("file1", help="Path to the first (gold-standard) CSV file.")
    compare_parser.add_argument("file2", help="Path to the second (predicted) CSV file.")
    compare_parser.add_argument("output_file", help="Path to the output text file.")

    # -------------------------------------------------------------- evolve
    evolve_parser = subparsers.add_parser(
        "evolve",
        help="Evolve a DSPy NER data-generation program with GEPA.",
    )
    evolve_parser.add_argument("train_csv", help="Path to the training CSV file.")
    evolve_parser.add_argument("dev_csv", help="Path to the dev/validation CSV file.")
    evolve_parser.add_argument(
        "--ollama-model", default="gemma3:27b",
        help="Ollama model name (default: gemma3:27b).",
    )
    evolve_parser.add_argument(
        "--ollama-base-url", default="http://localhost:11434",
        help="Ollama API base URL.",
    )
    evolve_parser.add_argument(
        "--max-metric-calls", type=int, default=100,
        help="GEPA budget: total evaluate() calls (default: 100).",
    )
    evolve_parser.add_argument(
        "--bert-epochs", type=int, default=5,
        help="BERT adapter training epochs per evaluation (default: 5).",
    )
    evolve_parser.add_argument(
        "--bert-batch-size", type=int, default=16,
        help="BERT batch size (default: 16).",
    )
    evolve_parser.add_argument(
        "--baseline-epochs", type=int, default=30,
        help="Epochs for baseline adapter training on real data (default: 30).",
    )
    evolve_parser.add_argument(
        "--run-dir", default="gepa_runs/ner_evolution",
        help="Directory for GEPA checkpoints/logs.",
    )

    # -------------------------------------------------------------- analyze
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze train and dev datasets and write an entity analysis report.",
    )
    analyze_parser.add_argument("trainset_file", help="Path to the training CSV file.")
    analyze_parser.add_argument("devset_file", help="Path to the dev/validation CSV file.")
    analyze_parser.add_argument("report_file", help="Path to the output text report file.")
    analyze_parser.add_argument(
        "--fuzzy-threshold",
        dest="fuzzy_threshold",
        type=float,
        default=0.5,
        help=(
            "Minimum Jaccard similarity (bag-of-words) to consider two entity "
            "strings a fuzzy match (default: 0.5)."
        ),
    )

    # ------------------------------------------------------------ evaluate
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run official SMM4H strict + relaxed NER evaluation on a CSV/Excel file.",
    )
    evaluate_parser.add_argument("input_file", help="Path to a CSV or Excel file with gold and predicted BIO tags.")
    evaluate_parser.add_argument(
        "--gold-col", default="ner_tags_str",
        help="Column name for gold BIO tags (default: ner_tags_str).",
    )
    evaluate_parser.add_argument(
        "--pred-col", default="prediction",
        help="Column name for predicted BIO tags (default: prediction).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "augment":
        augment(args.input_file, args.output_file, args.strategy, args.model)
        return

    if args.command == "analyze":
        from data.analyzer import FUZZY_THRESHOLD, build_report
        threshold = getattr(args, "fuzzy_threshold", FUZZY_THRESHOLD)
        report = build_report(args.trainset_file, args.devset_file, threshold=threshold)
        with open(args.report_file, "w", encoding="utf-8") as fh:
            fh.write(report)
            fh.write("\n")
        print(f"[analyze] Report written to {args.report_file}")
        return

    if args.command == "compare":
        compare(args.file1, args.file2, args.output_file)
        return

    if args.command == "evaluate":
        from data.evaluation import (
            _to_list,
            evaluate_test_strict_ner,
            calculate_f1_per_entity_covering_all,
        )
        import pandas as pd

        path = args.input_file
        if path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)

        strict = evaluate_test_strict_ner(
            df, gold_col=args.gold_col, pred_col=args.pred_col, print_report=True,
        )

        gold_tags = df[args.gold_col].apply(_to_list).tolist()
        pred_tags = df[args.pred_col].apply(_to_list).tolist()
        relaxed = calculate_f1_per_entity_covering_all(gold_tags, pred_tags)

        print("\n=== RELAXED OVERLAP NER METRICS ===")
        for entity_type, metrics in relaxed.items():
            print(f"Entity Type: {entity_type}")
            for k, v in metrics.items():
                print(f"  {k}: {v}")
            print()

        print("=" * 50)
        print(f"Micro F1 (Strict):  {strict['f1_strict']:.4f}")
        print(f"Overall F1 (Relaxed): {relaxed['Overall']['F1-Score']:.4f}")
        print("=" * 50)
        return

    if args.command == "evolve":
        from gepa_ner.run import run_evolution

        run_evolution(
            args.train_csv,
            args.dev_csv,
            ollama_model=args.ollama_model,
            ollama_base_url=args.ollama_base_url,
            max_metric_calls=args.max_metric_calls,
            bert_epochs=args.bert_epochs,
            bert_batch_size=args.bert_batch_size,
            baseline_epochs=args.baseline_epochs,
            run_dir=args.run_dir,
        )
        return

    trainer = Trainer(model_type=args.model_type)

    if args.command == "train":
        extra_kwargs: dict = {}
        if getattr(args, "ollama_model", None) is not None:
            extra_kwargs["ollama_model"] = args.ollama_model
        if getattr(args, "ollama_base_url", None) is not None:
            extra_kwargs["ollama_base_url"] = args.ollama_base_url
        if getattr(args, "max_metric_calls", None) is not None:
            extra_kwargs["max_metric_calls"] = args.max_metric_calls
        if getattr(args, "eval_sample_size", None) is not None:
            extra_kwargs["eval_sample_size"] = args.eval_sample_size
        trainer.train(
            args.input_file,
            args.validation_file,
            args.output_model,
            metric=args.metric,
            unlabeled_file=getattr(args, "unlabeled_file", None),
            word2vec_file=getattr(args, "word2vec_file", None),
            alpha_ce=getattr(args, "alpha_ce", 0.5),
            alpha_dice=getattr(args, "alpha_dice", 0.5),
            max_training_epochs=getattr(args, "max_training_epochs", 50),
            **extra_kwargs,
        )
    elif args.command == "test":
        test_kwargs: dict = {}
        if getattr(args, "ollama_model", None) is not None:
            test_kwargs["ollama_model"] = args.ollama_model
        if getattr(args, "ollama_base_url", None) is not None:
            test_kwargs["ollama_base_url"] = args.ollama_base_url
        trainer.test(
            args.input_file,
            args.output_file,
            getattr(args, "model_file", None),
            **test_kwargs,
        )


if __name__ == "__main__":
    main()
