LAZY_MODE = False
TENSOR_COUNTER = 0

class TensorOperations:
    @staticmethod
    def add(tensor1, tensor2):
        def grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.backward(grad)
            if tensor2.requires_grad:
                tensor2.backward(grad)

        tensor_data = tensor1.tensor_data + tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def scalar_add(tensor1, value):
        def grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.backward(grad)

        tensor_data = tensor1.tensor_data + value
        return tensor_data, grad_fn

    @staticmethod
    def sub(tensor1, tensor2):
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.backward(grad)
            if tensor2.requires_grad:
                tensor2_grad = (-1 * grad)
                tensor2.backward(tensor2_grad)

        tensor_data = tensor1.tensor_data - tensor2.tensor_data
        return tensor_data, _grad_fn

    @staticmethod
    def scalar_sub(tensor1, value):
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.backward(grad)
        tensor_data = tensor1.tensor_data - value
        return tensor_data, _grad_fn

    @staticmethod
    def mul(tensor1, tensor2):
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1_grad = (tensor2.tensor_data * grad)
                tensor1.backward(tensor1_grad)
            if tensor2.requires_grad:
                tensor2_grad = (tensor1.tensor_data * grad)
                tensor2.backward(tensor2_grad)

        tensor_data = tensor1.tensor_data * tensor2.tensor_data
        return tensor_data, _grad_fn

    @staticmethod
    def scalar_mul(tensor1, value):
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.grad = (grad * value)

        tensor_data = tensor1.tensor_data * value
        return tensor_data, _grad_fn

    @staticmethod
    def div(tensor1, tensor2):
        def _grad_fn(grad):
            # da(A/B) = 1/B
            if tensor1.requires_grad:
                tensor1_grad = (grad / tensor2.tensor_data)
                tensor1.backward(tensor1_grad)
            # db(A/B) = -A/B^2
            if tensor2.requires_grad:
                tensor2_grad = (-grad * tensor1.tensor_data) / (tensor2.tensor_data ** 2)
                tensor2.backward(tensor2_grad)

        tensor_data = tensor1.tensor_data / tensor2.tensor_data
        return tensor_data, _grad_fn

    @staticmethod
    def scalar_div(tensor1, value):
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.backward(grad / value)
        tensor_data = tensor1.tensor_data / value
        return tensor_data, _grad_fn

    @staticmethod
    def exp(tensor1, tensor2):
        def _grad_fn(grad):
            # dx(x^y) = y * x^(y-1)
            if tensor1.requires_grad:
                tensor1_grad = (grad * tensor2.tensor_data * (tensor1.tensor_data ** tensor2.tensor_data))
                tensor1.backward(tensor1_grad)
            # dy(x^y) = dy(e^(y*lnx)) = lnx*e^(y*lnx) = lnx * x^y
            if tensor2.requires_grad:
                # TODO: pass because im missing operations needed for this
                pass

        tensor_data = tensor1.tensor_data ** tensor2.tensor_data
        return tensor_data, _grad_fn


    @staticmethod
    def scalar_exp(tensor1, value):
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1_grad = (grad * (value * tensor1.tensor_data ** (value-1)))
                tensor1.grad(tensor1_grad)

        tensor_data = tensor1.tensor_data ** value
        return tensor_data, _grad_fn

    @staticmethod
    def scalar_log(tensor1):
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.backward(grad / tensor1.tensor_data)

        tensor_data = tensor1.tensor_data.log()
        return tensor_data, _grad_fn

    # TODO: should be transposed
    @staticmethod
    def matmul(tensor1, tensor2):
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1_grad = (tensor2.grad @ grad)
                tensor1.backward (tensor1_grad)
            if tensor2.requires_grad:
                tensor2_grad = (grad @ tensor1.grad)
                tensor2.backward(tensor2_grad)

        tensor_data = tensor1.tensor_data @ tensor2.tensor_data
        return tensor_data, _grad_fn

    @staticmethod
    def sum(tensor1, axes, keep_dims):
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1_grad = grad.broadcast(tensor1.tensor_data.shape())
                tensor1.backward(tensor1_grad)

        tensor_data = tensor1.tensor_data.sum(axes, keep_dims)
        return tensor_data, _grad_fn

    # destructive so we only send grad_fn back
    @staticmethod
    def swap(tensor1, axis1, axis2):
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.grad(grad.swap(axis1, axis2))

        tensor1.tensor_data.swap(axis1, axis2)
        return _grad_fn

    # destructive operation
    @staticmethod
    def broadcast(tensor1, new_shape):
        def _grad_fn(grad):
            if tensor1.requires_grad:
                sum_dims = []
                for i, (ts, ns) in enumerate(zip(tensor1.shape(), new_shape)):
                    if ts != ns:
                        sum_dims.append(i)
                tensor1_grad = grad.ones_like().sum(sum_dims)
                tensor1.backward(tensor1_grad)

        tensor1.broadcast(new_shape)
        return _grad_fn

    @staticmethod
    def _log_before_grad(op, grad, tensor1, tensor2):
        # Log information about the incoming gradient
        print(f"[{op} BACKWARD] Gradient shape: {grad.shape() if hasattr(grad, 'shape') else 'scalar'}")
        print(f"[{op} BACKWARD] Gradient value: {grad.data() if hasattr(grad, 'data') else grad}")

        # Log information about the tensors being added
        t1_name = tensor1._debug_name() or "unnamed_tensor1"
        t2_name = tensor2._debug_name() or "unnamed_tensor2"
        print(f"[{op} BACKWARD] Tensor1 '{t1_name}' shape: {tensor1.shape()}, requires_grad: {tensor1.requires_grad}")
        print(f"[{op} BACKWARD] Tensor2 '{t2_name}' shape: {tensor2.shape()}, requires_grad: {tensor2.requires_grad}")

        # Log the current gradient values of both tensors before update
        print(f"[{op} BACKWARD] Tensor1 '{t1_name}' grad before: {tensor1.grad.data() if tensor1.grad is not None else None}")
        print(f"[{op} BACKWARD] Tensor2 '{t2_name}' grad before: {tensor2.grad.data() if tensor2.grad is not None else None}")

    @staticmethod
    def _log_after_grad(op, tensor1, tensor2):
        t1_name = tensor1._debug_name() or "unnamed_tensor1"
        t2_name = tensor2._debug_name() or "unnamed_tensor2"
        print(f"[{op} BACKWARD] Tensor1 '{t1_name}' grad after: {tensor1.grad.data() if tensor1.grad is not None else None}")
        print(f"[{op} BACKWARD] Tensor2 '{t2_name}' grad after: {tensor2.grad.data() if tensor2.grad is not None else None}")