from core.labelers import LABELER_REGISTRY

def test_default_labeler():
    l = LABELER_REGISTRY.get("DefaultLabeler")(
        keyphrase_ngram_range=(1, 3), 
        threshold=0.5, 
        top_n=10
    )
    x = """
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
    y = l.forward(x)
    print(y)
