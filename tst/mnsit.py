import cProfile
import pstats

from MetalNeedle.loader import MnistDataLoader
from MetalNeedle.nn import Linear, ReLU, Softmax

# Input (784) -> Linear (784->128) -> ReLU -> Linear (128->10) -> Softmax
def mnist():
    data = MnistDataLoader("data/mnist/test-images", "data/mnist/test-labels", 10000)
    layer = Linear(784, 128)
    l1 = layer.forward(data.images)
    relu = ReLU()
    l2 = relu.forward(l1)
    layer2 = Linear(128, 10)
    l3 = layer2.forward(l2)
    softmax = Softmax()
    output = softmax.forward(l3)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    mnist()
    profiler.disable()
    # Print sorted stats
    stats = pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(10)