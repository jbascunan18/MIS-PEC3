from processing.preprocessor import Preprocessor
from processing.tagger import Tagger
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

tqdm.pandas()


def generate_processed_data(
    base_dir: Path, raw_dataset_path: Path, target_file: str, extra_processing=False
):
    preprocessor = Preprocessor()
    tagger = Tagger(
        gaming=(f"{base_dir}/data/gaming.txt"),
        technology=(f"{base_dir}/data/technology.txt"),
        extra_normalization=extra_processing,
    )

    dataset_df = pd.read_csv(
        raw_dataset_path,
        sep="\t",
    )
    dataset_df = dataset_df.dropna(subset=["text"])

    preprocessing_func = lambda x: preprocessor.preprocess_text(
        x, tilde_normalization=False, lemmatization=False
    )
    if extra_processing:
        preprocessing_func = lambda x: preprocessor.preprocess_text(
            x, tilde_normalization=True, lemmatization=True
        )

    dataset_df["preprocessed_text"] = dataset_df["text"].progress_apply(
        preprocessing_func
    )

    dataset_processed_df = dataset_df.dropna(subset=["preprocessed_text"])[
        ["preprocessed_text", "video_title", "category"]
    ].copy()

    dataset_processed_df[["category_processed", "category_score"]] = (
        dataset_processed_df.progress_apply(
            lambda row: tagger.tag(row["preprocessed_text"], row["video_title"]),
            axis=1,
            result_type="expand",
        )
    )

    dataset_processed_df.to_csv(
        f"{base_dir}/{target_file}",
        mode="a",
        header=not os.path.exists(f"{base_dir}/{target_file}"),
        sep="\t",
        index=False,
    )


def generate_train_test_split(
    base_dir: Path,
    preprocessed_dataset_path: Path,
    train_dataset_path: Path,
    test_dataset_path: Path,
):
    dataset_df = pd.read_csv(f"{base_dir}/{preprocessed_dataset_path}", sep="\t")
    tech_df = dataset_df[dataset_df["category_processed"] == "technology"].nlargest(
        1500, "category_score"
    )
    gaming_df = dataset_df[dataset_df["category_processed"] == "gaming"].nlargest(
        1500, "category_score"
    )

    merged_df = pd.concat([tech_df, gaming_df], ignore_index=True)
    final_df = merged_df[["preprocessed_text", "category_processed"]]

    train_df, test_df = train_test_split(
        final_df,
        stratify=final_df["category_processed"],
        test_size=0.2,
        random_state=42,
    )

    train_df.to_csv(
        f"{base_dir}/{train_dataset_path}",
        mode="a",
        header=not os.path.exists(f"{base_dir}/{train_dataset_path}"),
        sep="\t",
        index=False,
    )

    test_df.to_csv(
        f"{base_dir}/{test_dataset_path}",
        mode="a",
        header=not os.path.exists(f"{base_dir}/{test_dataset_path}"),
        sep="\t",
        index=False,
    )
