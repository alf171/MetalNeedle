

class Loss:
    """
    Base class for loss functions
    """
    def __init__(self, reduction = "mean"):
        # reduction method to use
        self.reduction = reduction

        # cache intermediate values
        self.cache = {}

    def __call__(self, predictions, targets):
        """
        Calculate the loss
        """
        raise NotImplemented("__call__ of Loss method not implemented")

    def backward(self, prediction, targets):
        """
        Calculate loss w.r.t prediction
        """
        pass

    def _reduce(self):
        """
        Apply the reduction method to a tensor
        """
        pass