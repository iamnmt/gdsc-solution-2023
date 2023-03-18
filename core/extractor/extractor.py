from keybert import KeyBERT

doc = ["""
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
      """,
      """
        Supervised learning, also known as supervised machine learning,
        is a subcategory of machine learning and artificial intelligence.
        It is defined by its use of labeled datasets to train algorithms that to classify data or predict outcomes accurately.
        As input data is fed into the model, it adjusts its weights until the model has been fitted appropriately,
        which occurs as part of the cross validation process. Supervised learning helps organizations solve for a variety of real-world problems at scale,
        such as classifying spam in a separate folder from your inbox.
      """]

class Extractor:
    def __init__(self, keyphrase_ngram_range: tuple, threshold: float) -> None:
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.model = KeyBERT()
        self.threshold = threshold
        self.keywords = []
        self.database = []

    def _process(self, docs: list) -> None:
        '''
            (Private) Process keyword extraction
            Input: Document as string
            Output: None, store keywords in keywords attribute
        '''
        keywords_list = self.model.extract_keywords(docs, keyphrase_ngram_range=(1, 3), stop_words='english')
        #i = 0 #debug
        for keywords in keywords_list:
            #print(f"Document #{i}: ")
            self.keywords = [word for word, prob in keywords if prob > self.threshold][:5]
            # Low diversity keyword handle
            if len(self.keywords) < 2:
                self.keywords = [word for word, prob in keywords][:2]
            # Add known keywords in database
            self.keywords += [word for word, prob in keywords if word in self.database]
            #print(f"Keywords: {self.keywords}")
            self.save_to_database()
            #i += 1
        
    def save_to_database(self) -> None:
        self.database += [keyword for keyword in self.keywords if keyword not in self.database]

    def get_keywords(self, doc: str) -> list:
        '''
            Return keywords for after-processing
            Input: Document as string
            Output: List of keywords.
        '''
        self._process(doc)
        return self.keywords

if __name__ == "__main__": 
    ex = Extractor(keyphrase_ngram_range=(1, 3), threshold=0.6)
    keywords = ex.get_keywords(doc=doc)
    print(keywords)