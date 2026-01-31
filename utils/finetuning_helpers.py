import numpy as np
import evaluate


def tokenize(tokenizer, examples):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy.compute(predictions=predictions, references=labels)
    prec = precision.compute(
        predictions=predictions, references=labels, average="weighted"
    )
    rec = recall.compute(predictions=predictions, references=labels, average="weighted")
    f1_score = f1.compute(
        predictions=predictions, references=labels, average="weighted"
    )
    return {
        "accuracy": acc["accuracy"],
        "precision": prec["precision"],
        "recall": rec["recall"],
        "f1": f1_score["f1"],
    }
