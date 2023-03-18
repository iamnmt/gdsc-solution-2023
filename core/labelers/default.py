from . import LABELER_REGISTRY
from keybert import KeyBERT


@LABELER_REGISTRY.register()
class DefaultLabeler:
    """
    Default labeler for the OCR model. 

    Uses KeyBERT to extract keywords from the document.
    """
    def __init__(self, keyphrase_ngram_range: list, threshold: float, top_n: int) -> None:
        """
        Input:
            keyphrase_ngram_range: The maximum n-gram size to consider.
            threshold: The minimum probability required for a keyword to be included.
            top_n: The number of keywords to extract.
        """
        self.model = KeyBERT()
        self.keyphrase_ngram_range = tuple(keyphrase_ngram_range)
        self.threshold = threshold
        self.top_n = top_n

    def forward(self, doc: str) -> list[str]:
        """
        Input:
            doc: The document to extract keywords from.
        Output:
            keywords: The extracted keywords.
        """
        keywords = self.model.extract_keywords(
            doc,
            keyphrase_ngram_range=self.keyphrase_ngram_range,
            stop_words="english",
            top_n=self.top_n
        )
        keywords = [word for word, prob in keywords if prob > self.threshold]
        return keywords
