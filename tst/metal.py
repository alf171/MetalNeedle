import MetalNeedle

def MetalAddTest():
    x = MetalNeedle.ones([32, 32], device="metal", dtype='float32', debug_name="x")
    y = MetalNeedle.ones([32, 32], device="metal", dtype='float32', debug_name="y")
    z = x + y
    assert(z[15,2] == 2)
    print("Metal add passed!")

if __name__ == "__main__":
    MetalAddTest()
