from MetalNeedle.nn import Linear, ReLU, Softmax
import MetalNeedle as mn

# Input (784) -> Linear (784→128) -> ReLU -> Linear (128→10) -> Softmax
def mnist():
    layer = Linear(784, 128)
    l1 = layer.forward(mn.randn([100, 784], dtype="float32"))
    relu = ReLU()
    l2 = relu.forward(l1)
    layer2 = Linear(128, 10)
    l3 = layer2.forward(l2)
    softmax = Softmax()
    output = softmax.forward(l3)
    print(output.sum()[0])

if __name__ == "__main__":
    mnist()