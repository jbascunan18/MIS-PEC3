import os

from processing.preprocessor import Preprocessor


class Tagger:
    def __init__(self, extra_normalization=False, **lexicon_paths):
        self.lexicons = {}
        self.preprocessor = Preprocessor()
        self.extra_normalization = extra_normalization

        self.initialize_lexicons(**lexicon_paths)

    def initialize_lexicons(self, **lexicon_paths):
        for label, path in lexicon_paths.items():
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    self.lexicons[label] = [
                        self.preprocessor.preprocess_text(
                            line.rstrip("\n"),
                            min_len=1,
                            tilde_normalization=self.extra_normalization,
                            lemmatization=self.extra_normalization,
                        )
                        for line in f.readlines()
                    ]
                    self.lexicons[label] = set(self.lexicons[label])

            else:
                print(f"Lexic file not existing for {label}")

    def compute_scores(self, text):
        tokens = set(text.split())
        scores = {label: 0 for label in self.lexicons.keys()}

        for label, lexicon_set in self.lexicons.items():
            matches = [t for t in tokens if t in lexicon_set]
            scores[label] += len(matches)

        return [s for s in scores.values()]

    def tag(self, preprocessed_comment: str, video_title: str) -> (str, int):
        tokens_comment = preprocessed_comment.split()

        preprocessed_video_title = self.preprocessor.preprocess_text(
            video_title, min_len=1
        )
        tokens_title = preprocessed_video_title.split()

        scores = {label: 0 for label in self.lexicons.keys()}

        for label, lexicon_set in self.lexicons.items():
            matches_comment = [t for t in tokens_comment if t in lexicon_set]
            scores[label] += len(matches_comment) * 2

            matches_title = [t for t in tokens_title if t in lexicon_set]
            scores[label] += len(matches_title)

        if max(scores.values()) == 0:
            return "Unknown", -1

        category_assigned = max(scores, key=scores.get)
        category_score = scores[category_assigned]

        return category_assigned, category_score
