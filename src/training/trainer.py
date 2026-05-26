"""Training and inference orchestrator for SMM4H NER models."""

import pandas as pd

from models.bert_adapter import BertAdapterModel
from models.bert_finetuned import BertFinetunedModel
from models.gliner_model import GlinerModel
from models.bert_variational import BertVariational
from models.bert_fixmatch import BertFixMatchModel
from models.bilstm import BiLSTMModel
from models.llm_ner_model import LLMNERModel

MODEL_REGISTRY = {
    "bert_adapter": BertAdapterModel,
    "bert_finetuned": BertFinetunedModel,
    "gliner": GlinerModel,
    "bert_variational": BertVariational,
    "bert_fixmatch": BertFixMatchModel,
    "lstm": BiLSTMModel,
    "llm_ner": LLMNERModel,
}


def load_data(file_path: str) -> pd.DataFrame:
    """Load and validate the task CSV file.

    The CSV is expected to have at least these columns:
    ``ID``, ``tokens``, ``labels``, ``ner_tags``.

    The ``tokens``, ``labels`` and ``ner_tags`` columns may be stored as
    string-encoded Python lists (e.g. ``"['Hello', ',', 'world']"``).
    """
    df = pd.read_csv(file_path)
    required_columns = {"ID", "tokens"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")
    return df


class Trainer:
    """Orchestrates training and inference across all supported model types."""

    def __init__(self, model_type: str):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Unsupported model type '{model_type}'. "
                f"Choose one of: {sorted(MODEL_REGISTRY.keys())}"
            )
        self.model_type = model_type
        self.model_class = MODEL_REGISTRY[model_type]

    # ------------------------------------------------------------------

    def train(self, input_file: str, validation_file: str, output_model: str, metric: str = "f1", unlabeled_file: str = None, word2vec_file: str = None, max_training_epochs: int = 50, alpha_ce: float = 0.5, alpha_dice: float = 0.5, **kwargs) -> None:
        """Load data, train the model, and save it to *output_model*.

        Parameters
        ----------
        input_file:
            Path to the training CSV.
        validation_file:
            Path to the validation CSV.
        output_model:
            Directory where the trained model will be saved.
        metric:
            Metric used for best-model selection (``"f1"``, ``"relaxed_f1"``, or
            ``"avg_f1"`` for the average of strict and relaxed F1).
        unlabeled_file:
            Optional path to unlabeled data for models that use it (e.g., fixmatch).
        word2vec_file:
            Optional path to a word2vec text-format file (used by the lstm model).
        max_training_epochs:
            Number of training epochs (default: 50).
        alpha_ce:
            Weight for the CrossEntropy loss component (default: 0.5).
        alpha_dice:
            Weight for the Dice loss component (default: 0.5).
        """
        print(f"[hner] Loading training data from: {input_file}")
        train_data = load_data(input_file)

        print(f"[hner] Loading validation data from: {validation_file}")
        val_data = load_data(validation_file)

        print(f"[hner] Training model type '{self.model_type}' (best-model metric: {metric})...")
        model = self.model_class()
        
        train_kwargs = {
            "metric": metric,
            "max_training_epochs": max_training_epochs,
            "alpha_ce": alpha_ce,
            "alpha_dice": alpha_dice,
        }
        if unlabeled_file is not None:
            train_kwargs["unlabeled_file"] = unlabeled_file
        if word2vec_file is not None:
            train_kwargs["word2vec_file"] = word2vec_file
        train_kwargs.update(kwargs)
        
        model.train(train_data, val_data, output_model, **train_kwargs)
        print(f"[hner] Model saved to: {output_model}")

    def test(self, input_file: str, output_file: str, model_file: str | None = None, **kwargs) -> None:
        """Load a model (or use zero-shot default), run inference, and write results.

        Parameters
        ----------
        input_file:
            Path to the test CSV.
        output_file:
            Path where the predictions CSV will be written.
        model_file:
            Optional path to a trained model directory.  When ``None``, a
            default zero-shot model is used.
        **kwargs:
            Extra keyword arguments forwarded to the model constructor when
            instantiating the default zero-shot model (e.g. ``ollama_model``
            or ``ollama_base_url`` for the ``llm_ner`` model type).
        """
        print(f"[hner] Loading test data from: {input_file}")
        test_data = load_data(input_file)

        if model_file:
            print(f"[hner] Loading model from: {model_file}")
            model = self.model_class.load(model_file)
        else:
            print(f"[hner] No model file specified — using default zero-shot model.")
            model = self.model_class(**kwargs)

        print(f"[hner] Running inference...")
        predictions = model.predict(test_data)

        predictions.to_csv(output_file, index=False)
        print(f"[hner] Predictions written to: {output_file}")
