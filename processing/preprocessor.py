import string
import nltk
import re
import emoji
import spacy
import unicodedata


class Preprocessor:
    def __init__(self):
        nltk.download("stopwords")
        nltk.download("punkt_tab")
        self.nlp = spacy.load("es_core_news_md")

    def preprocess_text(
        self, text: str, min_len: int = 3, tilde_normalization=True, lemmatization=True
    ) -> str:
        try:
            filtered_text = text
            if tilde_normalization:
                # tilde elimination
                filtered_text = unicodedata.normalize("NFD", filtered_text)
                filtered_text = "".join(
                    c for c in filtered_text if unicodedata.category(c) != "Mn"
                )
                filtered_text = unicodedata.normalize("NFC", filtered_text)

            # noise elimination: url, mentions, emojis
            filtered_text = re.sub(r"https?://\S+|www\.\S+", "", filtered_text)
            filtered_text = re.sub(r"@[\w]+", "", filtered_text)
            filtered_text = emoji.replace_emoji(filtered_text, replace="")

            if (
                filtered_text == ""
                or all(char in string.punctuation for char in filtered_text)
                or filtered_text.isnumeric()
            ):  # only data in text was removed or punctuation-only or number-only
                return None

            # normalization
            if lemmatization:
                doc = self.nlp(filtered_text)
                lemmatized_tokens = []
                for token in doc:
                    if token.is_punct or token.is_space:
                        continue
                    lemmatized_tokens.append(token.lemma_.lower())

                filtered_text = " ".join(lemmatized_tokens)
            filtered_text = filtered_text.lower()

            # stopwords elimination
            stops = set(nltk.corpus.stopwords.words("spanish"))
            filtered_text = " ".join(
                [w for w in filtered_text.split() if not w in stops]
            )

            # tokenization
            tokenized_text = nltk.tokenize.WhitespaceTokenizer().tokenize(filtered_text)

            # too short comment elimination
            if len(tokenized_text) < min_len:
                return None

            final_text = " ".join(tokenized_text)

        except Exception as e:
            print(f"Error while processing {text}: {e}")
            return None

        return final_text
