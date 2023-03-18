from keybert import KeyBERT

doc = """
         Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal).
         A supervised learning algorithm analyzes the training data and produces an inferred function,
         which can be used for mapping new examples. An optimal scenario will allow for the
         algorithm to correctly determine the class labels for unseen instances. This requires
         the learning algorithm to generalize from the training data to unseen situations in a
         'reasonable' way (see inductive bias).
      """

class Extractor:
    def __init__(self, keyphrase_ngram_range: tuple, threshold: float) -> None:
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.model = KeyBERT()
        self.threshold = threshold
        self.keywords = []

    def _process(self, doc: str) -> None:
        '''
            (Private) Process keyword extraction
            Input: Document as string
            Output: None, store keywords in keywords attribute
        '''
        keywords = self.model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), stop_words='english')
        self.keywords = [word for word, prob in keywords if prob > self.threshold][:5]
        if len(self.keywords) == 0:
            self.keywords = [word for word, prob in keywords][:2]
        
    def get_keywords(self, doc: str) -> list:
        '''
            Return keywords for after-processing
            Input: Document as string
            Output: List of keywords.
        '''
        self._process(doc)
        return self.keywords
    
ex = Extractor(keyphrase_ngram_range=(1, 3), threshold = 0.6)
keywords = ex.get_keywords(doc=doc)
print(keywords)
    

