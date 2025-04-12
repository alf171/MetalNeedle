from MetalNeedle.nn import Linear, ReLU
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

if __name__ == "__main__":
    linear()
    relu()