import numpy as np
from core.preprocess import PREPROCESS_REGISTRY

def test_default_preprocessor():
    p = PREPROCESS_REGISTRY.get("DefaultPreprocessor")()
    x = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    y = p.forward(x)
    print(y.shape)
