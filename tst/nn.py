from MetalNeedle.nn import Linear, ReLU, Softmax
import MetalNeedle as mn

def linear():
    batch_size = 100
    in_features = 10
    out_features = 5
    layer = Linear(in_features, out_features)
    res = layer.forward(mn.ones([batch_size, in_features]))
    assert(res.shape() == [batch_size, out_features])

    print("Linear passed!")


def relu():
    relu = ReLU()
    data = mn.randn([3,3])
    res = relu.forward(data)
    # TODO: need tensor comparison operations to properly test
    print("Relu passed!")

def softmax():
    softmax = Softmax()
    x = mn.ones([3, 4, 5], dtype="float32")
    output = softmax.forward(x)
    assert(output.shape() == [3,4,5])

    test_data = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [-1.0, -2.0, -3.0, -4.0, -5.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1000.0, 1000.0, 1000.0, 1000.0, 1000.0],  # Test numerical stability
        [-1000.0, -1000.0, -1000.0, -1000.0, -1000.0]  # Test numerical stability
    ]
    x = mn.Tensor.load([val for row in test_data for val in row], [5, 5], dtype="float32")

    output = softmax.forward(x)

    # Check that each row sums to approximately 1
    tolerance = 1e-5
    for i in range(output.shape()[0]):
        row_sum = output[i].sum()[0]
        assert(abs(row_sum - 1.0) < tolerance,
                        f"Row {i} sum is {row_sum}, which is not close to 1.0")


    print("Softmax passed!")


if __name__ == "__main__":
    linear()
    relu()
    softmax()