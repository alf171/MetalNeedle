LAZY_MODE = False
TENSOR_COUNTER = 0

class TensorOperations:
    @staticmethod
    def add(tensor1: 'Tensor', tensor2: 'Tensor'):
        def _backward(grad):
            # TensorOperations._log_before_grad("ADD", grad, tensor1, tensor2)
            if tensor1.requires_grad:
                tensor1.grad = grad + (tensor1.grad or 0)
            # else condition handles identical tensors being added
            if tensor2.requires_grad:
                tensor2.grad = grad + (tensor2.grad or 0)
            # TensorOperations._log_after_grad("ADD", tensor1, tensor2)

        tensorData = tensor1.tensorData + tensor2.tensorData
        return (tensorData, _backward)

    @staticmethod
    def scalar_add(tensor1, value):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = grad + (tensor1.grad or 0)

        tensorData = tensor1.tensorData + value
        return (tensorData, _backward)

    @staticmethod
    def sub(tensor1, tensor2):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = grad + (tensor1.grad or 0)
            if tensor2.requires_grad:
                tensor2.grad = (-1 * grad) + (tensor2.grad or 0)

        tensorData  = tensor1.tensorData - tensor2.tensorData
        return (tensorData, _backward)

    @staticmethod
    def scalar_sub(tensor1, value):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = grad + (tensor1.grad or 0)
        tensorData = tensor1.tensorData - value
        return (tensorData, _backward)

    @staticmethod
    def mul(tensor1, tensor2):
        def _backward(grad):
            # TensorOperations._log_before_grad("MUL", grad, tensor1, tensor2)
            if tensor1.requires_grad:
                tensor1.grad = (tensor2.tensorData * grad) #+ (tensor1.grad or 0)
            # else condition handles identical tensors being multiplied
            if tensor2.requires_grad:
                tensor2.grad = (tensor1.tensorData * grad.clone()) #+ (tensor2.grad or 0)
            # TensorOperations._log_after_grad("MUL", tensor1, tensor2)

        tensorData = tensor1.tensorData * tensor2.tensorData
        return (tensorData, _backward)

    @staticmethod
    def scalar_mul(tensor1, value):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = (grad * value) + (tensor1.grad or 0)

        tensorData = tensor1.tensorData * value
        return (tensorData, _backward)

    @staticmethod
    def div(tensor1, tensor2):
        def _backward(grad):
            # da(A/B) = 1/B
            if tensor1.requires_grad:
                tensor1.grad = (grad / tensor2.tensorData) + (tensor1.grad or 0)
            # db(A/B) = -A/B^2
            if tensor2.requires_grad:
                tensor2.grad = (-grad * tensor1.tensorData) / (tensor2.tensorData ** 2) + (tensor1.grad or 0)

        tensorData = tensor1.tensorData / tensor2.tensorData
        return (tensorData, _backward)

    @staticmethod
    def scalar_div(tensor1, value):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = (grad / value) + (tensor1.grad or 0)
        tensorData = tensor1.tensorData / value
        return (tensorData, _backward)

    @staticmethod
    def exp(tensor1, tensor2):
        def _backward(grad):
            # dx(x^y) = y * x^(y-1)
            if tensor1.requires_grad:
                tensor1.grad = (grad * tensor2.tensorData * (tensor1.tensorData ** tensor2.tensorData)) + (tensor1.grad or 0)
            # dy(x^y) = dy(e^(y*lnx)) = lnx*e^(y*lnx) = lnx * x^y
            if tensor2.requires_grad:
                # TODO: pass because im missing operations needed for this
                pass

        tensorData = tensor1.tensorData ** tensor2.tensorData
        return (tensorData, _backward)


    @staticmethod
    def scalar_exp(tensor1, value):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = (grad * (value * tensor1.tensorData ** (value-1))) + (tensor1.grad or 0)

        tensorData = tensor1.tensorData ** value
        return (tensorData, _backward)

    @staticmethod
    def scalar_log(tensor1):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = (grad / tensor1.tensorData) + (tensor1.grad or 0)

        tensorData = tensor1.tensorData.log()
        return (tensorData, _backward)

    # TODO: should be transposed
    @staticmethod
    def matmul(tensor1, tensor2):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = (tensor2.grad @ grad) + (tensor1.grad + 0)
            if tensor2.requires_grad:
                tensor1.grad = (grad @ tensor1.grad) + (tensor1.grad + 0)

        tensorData = tensor1.tensorData @ tensor2.tensorData
        return (tensorData, _backward)

    @staticmethod
    def sum(tensor1, axes, keepDims):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = grad.broadcast(tensor1.tensorData.shape()) + (tensor1.grad or 0)

        tensorData = tensor1.tensorData.sum(axes, keepDims)
        return (tensorData, _backward)

    # destructive so we only send _backwards back
    @staticmethod
    def swap(tensor1, axis1, axis2):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = grad.swap(axis1, axis2) + (tensor1.grad or 0)

        tensor1.swap(axis1, axis2)
        return _backward

    @staticmethod
    # destructive operation
    def broadcast(tensor1, newShape):
        def _backward(grad):
            if tensor1.requires_grad:
                sum_dims = []
                for i, (ts, ns) in enumerate(zip(tensor1.shape(), newShape)):
                    if ts != ns:
                        sum_dims.append(i)
                tensor1.grad = grad.ones_like().sum(sum_dims) + (tensor1.grad or 0)

        tensor1.broadcast(newShape)
        return _backward

    @staticmethod
    def _log_before_grad(op, grad, tensor1, tensor2):
        # Log information about the incoming gradient
        print(f"[{op} BACKWARD] Gradient shape: {grad.shape() if hasattr(grad, 'shape') else 'scalar'}")
        print(f"[{op} BACKWARD] Gradient value: {grad.data() if hasattr(grad, 'data') else grad}")

        # Log information about the tensors being added
        t1_name = tensor1._debug_name or "unnamed_tensor1"
        t2_name = tensor2._debug_name or "unnamed_tensor2"
        print(f"[{op} BACKWARD] Tensor1 '{t1_name}' shape: {tensor1.shape()}, requires_grad: {tensor1.requires_grad}")
        print(f"[{op} BACKWARD] Tensor2 '{t2_name}' shape: {tensor2.shape()}, requires_grad: {tensor2.requires_grad}")

        # Log the current gradient values of both tensors before update
        print(f"[{op} BACKWARD] Tensor1 '{t1_name}' grad before: {tensor1.grad.data() if tensor1.grad is not None else None}")
        print(f"[{op} BACKWARD] Tensor2 '{t2_name}' grad before: {tensor2.grad.data() if tensor2.grad is not None else None}")

    @staticmethod
    def _log_after_grad(op, tensor1, tensor2):
        t1_name = tensor1._debug_name or "unnamed_tensor1"
        t2_name = tensor2._debug_name or "unnamed_tensor2"
        print(f"[{op} BACKWARD] Tensor1 '{t1_name}' grad after: {tensor1.grad.data() if tensor1.grad is not None else None}")
        print(f"[{op} BACKWARD] Tensor2 '{t2_name}' grad after: {tensor2.grad.data() if tensor2.grad is not None else None}")