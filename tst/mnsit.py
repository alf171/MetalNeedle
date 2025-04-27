import cProfile
import pstats

from MetalNeedle.loader import MnistDataLoader
from MetalNeedle.loss.cat_cross_entropy import CategoricalCrossEntropy
from MetalNeedle.nn import Linear, ReLU, Softmax
from MetalNeedle.optim.sgd import SGD


# Input (784) -> Linear (784->128) -> ReLU -> Linear (128->10) -> Softmax
def mnist() -> None:
    data = MnistDataLoader("data/mnist/test-images", "data/mnist/test-labels", 1000)
    layer1 = Linear(784, 128)
    relu = ReLU()
    layer2 = Linear(128, 10)
    softmax = Softmax()
    loss_fn = CategoricalCrossEntropy()

    # optim step
    parameters = [layer1.weight, layer1.bias, layer2.weight, layer2.bias]
    optimizer = SGD(parameters, lr=0.01, momentum=0.9)

    num_batches = data.images.numel() // data.batch_size
    for i in range(100):
        l1 = layer1.forward(data.images)
        l2 = relu.forward(l1)
        l3 = layer2.forward(l2)
        output = softmax.forward(l3)
        loss_value = loss_fn(output, data.labels)

        # backward pass
        output.backward()

        # optim
        optimizer.zero_grad()
        optimizer.step()

if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    mnist()
    # profiler.disable()
    # # Print sorted stats
    # stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE)
    # stats.print_stats(10)