from core.preprocess import PREPROCESS

def test_add_funcition():
    add_func = PREPROCESS.get("add")
    assert add_func(5, 6) == 5+6+1
