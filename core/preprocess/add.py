from . import PREPROCESS

@PREPROCESS.register()
def add(x, y):
    return x+y
