from .loss import Loss
from .. import Tensor


# https://www.geeksforgeeks.org/categorical-cross-entropy-in-multi-class-classification/
class CategoricalCrossEntropy(Loss):
    def __init__(self, reduction="mean", epsilon=1e-7):
        super().__init__(reduction)
        self.epsilon = epsilon

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # clip for numeric stability
        safe_predictions = predictions.clip(self.epsilon, 1 - self.epsilon)

        # negative log likelihood
        loss = -targets * safe_predictions.log()

        return self._reduce(loss)

