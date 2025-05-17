from MetalNeedle.nn import Linear, ReLU, Softmax
import MetalNeedle as mn
import numpy as np

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
    x = mn.Tensor.load_from_buffer([val for row in test_data for val in row], [5, 5], dtype="float32")

    tolerance = 1e-5
    output = softmax.forward(x)
    for i in range(output.shape()[0]):
        row_sum = output.sum(axes=-1)[i]
        assert(abs(row_sum - 1.0) < tolerance)

    test_input = [2.0, 1.0, 0.0]
    x = mn.Tensor.load_from_buffer(test_input, [1, 3], dtype="float32")
    output = softmax.forward(x)

    # Calculate expected values using numpy for comparison
    np_input = np.array(test_input)
    exp_input = np.exp(np_input - np.max(np_input))
    expected = exp_input / np.sum(exp_input)

    # Check values are close to expected
    tolerance = 1e-5
    for i in range(output.shape()[1]):
        assert(abs(output[0, i] - expected[i]) < tolerance)


    print("Softmax passed!")


if __name__ == "__main__":
    linear()
    relu()
    softmax()