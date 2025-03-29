from src.MetalNeedle import Tensor


# this is our base module class all other neural network
# modules will piggyback off of this one
class Module:
    def __init__(self):
        # store tensors
        self._parameters = {}
        # store other modules
        self._modules = {}
        # indicate what mode we are in
        self.training = True

    def register_parameter(self, name, tensor):
        """
        register a parameter
        """
        self._parameters[name] = tensor

    def register_module(self, name, module):
        """
        store modules with our base class
        """
        self._modules[name] = module

    def parameters(self):
        """
        return all parameters in the class and submodules
        """
        params = []
        for param in self._parameters.values():
            params.append(param)

        for module in self._modules.values():
            params.extend(module._parameters.values())

        return params

    def set_training_mode(self, mode=True):
        """
        Set the training mode
        """
        self.training = mode

    def __setattr__(self, name, value):
        """
        When a module is defined, we want to automatically
        register all parts
        """
        if isinstance(value, Tensor):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            raise ValueError(f"invalid module type {type(value)}")


    def forward(self, *args, **kwargs):
        """
        inheritor must implement a custom forward for their module
        """
        raise NotImplemented("forward pass not implemented")

    def __call__(self, *args, **kwargs):
        self.forward(args, kwargs)
