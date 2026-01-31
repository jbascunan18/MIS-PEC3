from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from processing.tagger import Tagger
from scipy.sparse import csr_matrix, hstack


class TextFeaturizer:

    def __init__(self, base_dir, extra_normalization=False):
        self.tfidf = TfidfVectorizer(norm="l2", max_features=1000)
        self.tagger = Tagger(
            gaming=f"{base_dir}/data/gaming.txt",
            technology=f"{base_dir}/data/technology.txt",
            extra_normalization=extra_normalization,
        )
        self.scaler = MinMaxScaler()

    def generate_tfidf(
        self,
        train_preprocessed_text_df: pd.core.series.Series,
        test_preprocessed_text_df: pd.core.series.Series,
    ):
        train_tfidf_features = self.tfidf.fit_transform(train_preprocessed_text_df)
        test_tfidf_features = self.tfidf.transform(test_preprocessed_text_df)

        return train_tfidf_features, test_tfidf_features

    def _compute_lexicons_score(self, train_preprocessed_text_df):
        all_scores = []
        for text in train_preprocessed_text_df:
            scores = self.tagger.compute_scores(text)
            all_scores.append(scores)

        return all_scores

    def generate_tfidf_extended(
        self,
        train_preprocessed_text_df: pd.core.series.Series,
        test_preprocessed_text_df: pd.core.series.Series,
    ):
        train_tfidf_features, test_tfidf_features = self.generate_tfidf(
            train_preprocessed_text_df, test_preprocessed_text_df
        )

        train_lexicons_scores = self._compute_lexicons_score(train_preprocessed_text_df)
        train_lexicons_scores = self.scaler.fit_transform(train_lexicons_scores)
        test_lexicons_scores = self._compute_lexicons_score(test_preprocessed_text_df)
        test_lexicons_scores = self.scaler.transform(test_lexicons_scores)

        train_lexicons_scores_sparse = csr_matrix(train_lexicons_scores)
        test_lexicons_scores_sparse = csr_matrix(test_lexicons_scores)

        train_final_representation = hstack(
            [train_tfidf_features, train_lexicons_scores_sparse]
        )
        test_final_representation = hstack(
            [test_tfidf_features, test_lexicons_scores_sparse]
        )

        return train_final_representation, test_final_representation
