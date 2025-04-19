from MetalNeedle import Tensor


class Loss:
    """
    Base class for loss functions
    """
    def __init__(self, reduction = "mean"):
        # reduction method to use
        self.reduction = reduction

        # cache intermediate values -- currently not used
        self.cache = {}

    def __call__(self, predictions, targets) -> None:
        """
        Calculate the loss
        """
        raise NotImplemented("[loss] __call__ of Loss method not implemented")

    def backward(self, prediction, targets) -> None:
        """
        Calculate loss w.r.t prediction
        """
        raise NotImplemented("[loss] backward not implemented")

    def _reduce(self, loss_tensor) -> Tensor:
        """
        Apply the reduction method to a tensor
        """
        if self.reduction == "mean":
            return loss_tensor.mean()
        elif self.reduction == "sum":
            return loss_tensor.sum()
        elif self.reduction == "none":
            return loss_tensor
        else:
            raise ValueError(f"[loss] unknown reduce method {self.reduction}")