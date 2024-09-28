from Needle.main import Needle
from Needle.init import *

# x = Needle.Tensor([[1, 2, 3], [4, 5, 6]])
# y = Needle.Tensor([[1, 2, 3], [4, 5, 6]])
# z = x * y
# print(z._data.data)

x = Needle.randn([3,3,1], 0, 1, "float32")
print(x)
# y = Needle.randn([3,3,3], 0, 1, "float32")
# print(y)
# z = x @ y
# print(x._tensor)