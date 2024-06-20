from Needle.main import Needle
from Needle.init import *

# x = Needle.Tensor([[1, 2, 3], [4, 5, 6]])
# y = Needle.Tensor([[1, 2, 3], [4, 5, 6]])
# z = x * y
# print(z._data.data)

x = Needle.randn([3,3], 10, 5, "float32")
print(x.get_item([0,0]))
