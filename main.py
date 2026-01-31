import pandas as pd
import numpy as np
import os
import gc
import torch
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    AutoTokenizer,
)
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset, DatasetDict

from processing.text_featurizer import TextFeaturizer
from utils.evaluation_helpers import (
    train_and_evaluate_classifier,
    generate_classification_report,
    plot_correlation_matrix,
)
from services.input_parser import InputParser
from scripts.prepare_dataset import generate_processed_data, generate_train_test_split
from utils.finetuning_helpers import tokenize, compute_metrics as ft_compute_metrics

BASE_DIR = Path(__file__).resolve().parent
RAW_DATASET_PATH = "data/dataset_raw.csv"
SVM_PROCESSED_DATASET = "target/dataset_processed_svm.csv"
SVM_TRAIN_DATASET = "target/dataset_svm_train.csv"
SVM_TEST_DATASET = "target/dataset_svm_test.csv"
TRANSFORMERS_PROCESSED_DATASET = "target/dataset_processed_transformers.csv"
TRANSFORMERS_TRAIN_DATASET = "target/dataset_transformers_train.csv"
TRANSFORMERS_TEST_DATASET = "target/dataset_transformers_test.csv"


def main():

    input_parser = InputParser()
    args = input_parser.parse_args()

    svm_train_data = None
    svm_test_data = None
    transformers_train_data = None
    transformers_test_data = None

    # Prepare datasets for both approaches (SVM + Transformers) if they don't exist yet
    if not os.path.exists(f"{BASE_DIR}/{SVM_PROCESSED_DATASET}"):
        generate_processed_data(
            BASE_DIR,
            RAW_DATASET_PATH,
            SVM_PROCESSED_DATASET,
            extra_processing=True,
        )

    if not os.path.exists(f"{BASE_DIR}/{SVM_TRAIN_DATASET}") or not os.path.exists(
        f"{BASE_DIR}/{SVM_TEST_DATASET}"
    ):
        generate_train_test_split(
            BASE_DIR, SVM_PROCESSED_DATASET, SVM_TRAIN_DATASET, SVM_TEST_DATASET
        )

    svm_train_data = pd.read_csv(f"{BASE_DIR}/{SVM_TRAIN_DATASET}", sep="\t")
    svm_test_data = pd.read_csv(f"{BASE_DIR}/{SVM_TEST_DATASET}", sep="\t")

    if not os.path.exists(f"{BASE_DIR}/{TRANSFORMERS_PROCESSED_DATASET}"):
        generate_processed_data(
            BASE_DIR,
            RAW_DATASET_PATH,
            TRANSFORMERS_PROCESSED_DATASET,
            extra_processing=False,
        )

    if not os.path.exists(
        f"{BASE_DIR}/{TRANSFORMERS_TRAIN_DATASET}"
    ) or not os.path.exists(f"{BASE_DIR}/{TRANSFORMERS_TEST_DATASET}"):
        generate_train_test_split(
            BASE_DIR,
            TRANSFORMERS_PROCESSED_DATASET,
            TRANSFORMERS_TRAIN_DATASET,
            TRANSFORMERS_TEST_DATASET,
        )

    transformers_train_data = pd.read_csv(
        f"{BASE_DIR}/{TRANSFORMERS_TRAIN_DATASET}", sep="\t"
    )
    transformers_test_data = pd.read_csv(
        f"{BASE_DIR}/{TRANSFORMERS_TEST_DATASET}", sep="\t"
    )
    transformers_validation_data, transformers_test_data = train_test_split(
        transformers_test_data, test_size=0.2, random_state=42
    )

    # Same encoding for both approaches (SVM + Transformers)
    label_encoder = LabelEncoder()
    category_values = (
        svm_train_data.category_processed.values
        if not svm_train_data is None
        else transformers_train_data.category_processed.values
    )
    label_encoder.fit(category_values)
    target_labels = label_encoder.classes_
    encoded_y_train = label_encoder.transform(category_values)
    text_featurizer = TextFeaturizer(BASE_DIR)

    models_results = []
    ####### SVM #######
    ## Case 1: Representacion TF-IDF Representation
    if "svm" in args.model or "all" in args.model:
        svm_train_tfidf_features, svm_test_tfidf_features = (
            text_featurizer.generate_tfidf(
                svm_train_data.preprocessed_text, svm_test_data.preprocessed_text
            )
        )

        svm_grid = [
            {"C": [1, 10, 100], "kernel": ["linear"]},
            {"C": [10, 100, 1000], "gamma": [0.0001], "kernel": ["rbf"]},
        ]

        svm_cls = train_and_evaluate_classifier(
            svm_train_tfidf_features, encoded_y_train, SVC(), svm_grid
        )
        svm_y_pred = svm_cls.predict(svm_test_tfidf_features)
        svm_y_pred = label_encoder.inverse_transform(svm_y_pred)

        svm_evaluation_data = (
            svm_y_pred,
            svm_test_data.category_processed,
            "SVM + TF-IDF",
        )
        models_results.append(svm_evaluation_data)

    ## Case 2: TF-IDF Representation + lexicon score
    if "svm_ext" in args.model or "all" in args.model:
        svm_ext_train_features, svm_ext_test_features = (
            text_featurizer.generate_tfidf_extended(
                svm_train_data.preprocessed_text, svm_test_data.preprocessed_text
            )
        )

        # Correlation matrix between lexicon features and label
        train_lexicon_scores = text_featurizer._compute_lexicons_score(
            svm_train_data.preprocessed_text
        )
        plot_correlation_matrix(
            train_lexicon_scores,
            encoded_y_train,
            title=" SVM Extended Lexicon Features vs Label Correlation",
        )

        svm_grid = [
            {"C": [1, 10, 100], "kernel": ["linear"]},
            {"C": [10, 100, 1000], "gamma": [0.0001], "kernel": ["rbf"]},
        ]

        svm_ext_cls = train_and_evaluate_classifier(
            svm_ext_train_features, encoded_y_train, SVC(), svm_grid
        )
        svm_ext_y_pred = svm_ext_cls.predict(svm_ext_test_features)
        svm_ext_y_pred = label_encoder.inverse_transform(svm_ext_y_pred)

        svm_ext_evaluation_data = (
            svm_ext_y_pred,
            svm_test_data.category_processed,
            "SVM Extended + TF-IDF + scores",
        )

        models_results.append(svm_ext_evaluation_data)

    ####### TRANSFORMERS #######
    ## Case 1: Zero-Shot (Inference on pretrained model)
    if "zs_transformers" in args.model or "all" in args.model:
        zero_shot_classifier = pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli", batch_size=16
        )
        labels = ["tecnologia hardware", "videojuegos gaming"]

        # Since it's pure inference, there's no need to split into train/test
        all_data_df = pd.concat([transformers_train_data, transformers_test_data])
        all_data_dataset = Dataset.from_pandas(all_data_df)
        predictions_it = zero_shot_classifier(
            KeyDataset(all_data_dataset, "preprocessed_text"), labels
        )
        predicted_labels = []
        for prediction in predictions_it:
            label = prediction["labels"][np.argmax(prediction["scores"])]
            label_mapped = "technology" if label == "tecnologia hardware" else "gaming"
            predicted_labels.append(label_mapped)

        zero_shot_pred_df = pd.DataFrame(predicted_labels)

        zero_shot_evaluation_data = (
            zero_shot_pred_df,
            all_data_df.category_processed,
            "Zero-Shot Transformers",
        )
        models_results.append(zero_shot_evaluation_data)

        del zero_shot_classifier
        gc.collect()
        torch.cuda.empty_cache()

    ## Case 2: Fine-tune
    if "ft_transformers" in args.model or "all" in args.model:
        ft_model_name = "FacebookAI/xlm-roberta-base"
        ft_model_tokenizer = AutoTokenizer.from_pretrained(ft_model_name)

        # Adapt to the format expected by the Trainer API (Hugging Face)
        transformers_train_data = transformers_train_data.rename(
            columns={"preprocessed_text": "text", "category_processed": "label"}
        )
        transformers_train_data["label"] = transformers_train_data["label"].apply(
            lambda x: 0 if x == "gaming" else 1
        )
        transformers_validation_data = transformers_validation_data.rename(
            columns={"preprocessed_text": "text", "category_processed": "label"}
        )
        transformers_validation_data["label"] = transformers_validation_data[
            "label"
        ].apply(lambda x: 0 if x == "gaming" else 1)
        transformers_test_data = transformers_test_data.rename(
            columns={"preprocessed_text": "text", "category_processed": "label"}
        )
        transformers_test_data["label"] = transformers_test_data["label"].apply(
            lambda x: 0 if x == "gaming" else 1
        )

        # Prepare 3 splits: train, validation and test (validation is needed for the Trainer API to evaluate during training and select the best model)
        ft_dataset_train = Dataset.from_pandas(
            transformers_train_data.reset_index(drop=True)
        )
        ft_dataset_validation = Dataset.from_pandas(
            transformers_validation_data.reset_index(drop=True)
        )
        ft_dataset_test = Dataset.from_pandas(
            transformers_test_data.reset_index(drop=True)
        )

        ft_dataset = DatasetDict(
            {
                "train": ft_dataset_train,
                "validation": ft_dataset_validation,
                "test": ft_dataset_test,
            }
        )

        # Dataset tokenization
        ft_dataset_tokenized = ft_dataset.map(
            lambda examples: tokenize(ft_model_tokenizer, examples), batched=True
        )
        ft_dataset_tokenized = ft_dataset_tokenized.map(
            lambda examples: {"labels": examples["label"]}, batched=True
        )

        ft_dataset_tokenized = ft_dataset_tokenized.select_columns(
            ["input_ids", "attention_mask", "labels"]
        )
        ft_dataset_tokenized.with_format("torch")

        ft_model = None
        do_train = True
        # Use the checkpoint if it exists to avoid retraining every time
        if os.path.exists(f"{BASE_DIR}/target/ft_model/checkpoint-600"):
            ft_model = AutoModelForSequenceClassification.from_pretrained(
                f"{BASE_DIR}/target/ft_model/checkpoint-600", num_labels=2
            )
            do_train = False

        else:  # Generate the model (doesn't exist yet)
            ft_model = AutoModelForSequenceClassification.from_pretrained(
                ft_model_name, num_labels=2
            )

        training_args = TrainingArguments(
            output_dir="target/ft_model",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            remove_unused_columns=False,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=ft_model,
            args=training_args,
            train_dataset=ft_dataset_tokenized["train"],
            eval_dataset=ft_dataset_tokenized["validation"],
            processing_class=ft_model_tokenizer,
            compute_metrics=ft_compute_metrics,
        )

        if do_train:
            trainer.train()

        metrics = trainer.evaluate()
        print(metrics)

        # Get predictions from the trainer on the test dataset
        predictions = trainer.predict(ft_dataset_tokenized["test"])

        # Extract true labels and predicted labels
        true_labels = list(
            map(lambda x: "gaming" if x == 0 else "technology", predictions.label_ids)
        )
        predicted_labels = list(
            map(
                lambda x: "gaming" if x == 0 else "technology",
                predictions.predictions.argmax(-1),
            )
        )

        ft_evaluation_data = (predicted_labels, true_labels, "Fine-tuned Transformers")
        models_results.append(ft_evaluation_data)

    # Show all evaluation results
    for evaluation_data in models_results:
        generate_classification_report(
            evaluation_data[0],
            evaluation_data[1],
            target_labels,
            title=evaluation_data[2],
        )


if __name__ == "__main__":
    main()
