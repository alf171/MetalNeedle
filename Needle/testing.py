from Needle.main import Needle
from Needle.init import *

# x = Needle.Tensor([[1, 2, 3], [4, 5, 6]])
# y = Needle.Tensor([[1, 2, 3], [4, 5, 6]])
# z = x * y
# print(z._data.data)

x = Needle.randn([8, 8], 0, 1, "float32")
y = Needle.randn([8, 8], 0, 1, "float32")
z = x @ y
print(z)