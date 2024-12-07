from src.needle.main import Needle
from src.needle.init import *

# x = Needle.Tensor([[1, 2, 3], [4, 5, 6]])
# y = Needle.Tensor([[1, 2, 3], [4, 5, 6]])
# z = x * y
# print(z._data.data)

x = Needle.ones([3, 3], "float32")
# print(x)
y = Needle.ones([3, 3], "float32")
# print(y)

z = (x @ y)
print(z)
# print(z.get_item([0,0]))