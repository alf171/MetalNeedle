from .loss import Loss
from .. import Tensor


# https://www.geeksforgeeks.org/categorical-cross-entropy-in-multi-class-classification/
class CategoricalCrossEntropy(Loss):
    def __init__(self, reduction="mean", epsilon=1e-7):
        super().__init__(reduction)
        self.epsilon = epsilon

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # negative log likelihood
        per_example = -(targets * (predictions + self.epsilon).log()).sum(axes=-1)

        return per_example.mean()
