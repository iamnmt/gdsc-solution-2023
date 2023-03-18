from keybert import KeyBERT

doc = '''Al Engineering Al Engineer job title will covers CareerPaths the domain-based job titles and platform-based job titles Al Researcher Al Knowledge Scientist Al Knowledge ScientistMachine Learning Engineer Microsoft AI/MLEngineer AWS AI/ML Engineer Al Scientist Applied MLEngineer Google AI/MLEngineer Al Robotics Engineer Deep Learning Engineer Facebook AI/ML Engineer Al Algorithm EngineerNaive Bayes Learning Engineer Alibaba AI/MLEngineer AIOps/MLOps EngineerNPL Learning Engineer BM Machine Learning Engineer AI Engineer Tensorflow Engineer AI Developer Scikit-Learn Engineer Dala Cloud laT Presented by HUU KHANGPHAM'''

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
        keywords= self.model.extract_keywords(docs, keyphrase_ngram_range=(1, 3), stop_words='english')
        #i = 0 #debug
        #print(keywords)
        self.keywords = [word for word, prob in keywords if prob > self.threshold]
            # Low diversity keyword handle
        if len(self.keywords) < 2:
            self.keywords = [word for word, _ in keywords]
            # Add known keywords in database
        self.keywords += [word for word, _ in keywords if word in self.database]
        #print(f"Keywords: {self.keywords}")
        self.save_to_database()
        
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
