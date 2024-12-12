from .init import *

def ThreeByThreeMatMulCheck():
    x = Needle.Tensor([[1,2,3],[4,5,6],[7,8,9]])
    y = Needle.Tensor([[1,2,3],[4,5,6],[7,8,9]])
    z = (x @ y)

    # 0th row
    assert(z.get_item([0,0]) == 30)
    assert(z.get_item([0,1]) == 36)
    assert(z.get_item([0,2]) == 42)

    # 1st row
    assert(z.get_item([1,0]) == 66)
    assert(z.get_item([1,1]) == 81)
    assert(z.get_item([1,2]) == 96)

    # 2nd row
    assert(z.get_item([2,0]) == 102)
    assert(z.get_item([2,1]) == 126)
    assert(z.get_item([2,2]) == 150)

ThreeByThreeMatMulCheck()