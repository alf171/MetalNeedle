import unittest

from MetalNeedle import Tensor
from MetalNeedle.nn.activations.relu import ReLU
from MetalNeedle.nn.linear import Linear


def _finite_difference(loss_fn, values, epsilon=1e-3):
    grads = []
    for i in range(len(values)):
        plus = values[:]
        minus = values[:]
        plus[i] += epsilon
        minus[i] -= epsilon
        grads.append((loss_fn(plus) - loss_fn(minus)) / (2 * epsilon))
    return grads


class TestGradOps(unittest.TestCase):
    def assert_grad_close(self, analytic, numerical, places=2):
        self.assertEqual(len(analytic), len(numerical))
        for analytic_grad, numerical_grad in zip(analytic, numerical):
            self.assertAlmostEqual(analytic_grad, numerical_grad, places=places)

    def test_matmul_gradient_matches_finite_difference(self):
        x_values = [1.0, -2.0, 0.5, 3.0]
        w_values = [0.2, -0.3, 1.1, 0.7]

        x = Tensor([[x_values[0], x_values[1]], [x_values[2], x_values[3]]], dtype="float32", requires_grad=True)
        w = Tensor([[w_values[0], w_values[1]], [w_values[2], w_values[3]]], dtype="float32", requires_grad=True)
        loss = (x @ w).sum()
        loss.backward()

        def loss_for_x(flat_values):
            tensor = Tensor([[flat_values[0], flat_values[1]], [flat_values[2], flat_values[3]]], dtype="float32")
            weight = Tensor([[w_values[0], w_values[1]], [w_values[2], w_values[3]]], dtype="float32")
            return (tensor @ weight).sum()[0]

        def loss_for_w(flat_values):
            tensor = Tensor([[x_values[0], x_values[1]], [x_values[2], x_values[3]]], dtype="float32")
            weight = Tensor([[flat_values[0], flat_values[1]], [flat_values[2], flat_values[3]]], dtype="float32")
            return (tensor @ weight).sum()[0]

        self.assert_grad_close(x.grad.data(), _finite_difference(loss_for_x, x_values))
        self.assert_grad_close(w.grad.data(), _finite_difference(loss_for_w, w_values))

    def test_relu_gradient_matches_finite_difference(self):
        x_values = [-1.5, 0.8, 2.2]
        x = Tensor([x_values], dtype="float32", requires_grad=True)
        loss = ReLU().forward(x).sum()
        loss.backward()

        def loss_for_x(flat_values):
            tensor = Tensor([flat_values], dtype="float32")
            return ReLU().forward(tensor).sum()[0]

        self.assert_grad_close(x.grad.data(), _finite_difference(loss_for_x, x_values))

    def test_linear_weight_and_bias_gradient_match_finite_difference(self):
        x_values = [1.0, -2.0]
        weight_values = [0.5, -0.25, 1.5, 0.75]
        bias_values = [0.1, -0.2]

        layer = Linear(2, 2)
        layer.weight = Tensor(
            [[weight_values[0], weight_values[1]], [weight_values[2], weight_values[3]]],
            dtype="float32",
            requires_grad=True,
        )
        layer.bias = Tensor([bias_values[0], bias_values[1]], dtype="float32", requires_grad=True)

        x = Tensor([x_values], dtype="float32", requires_grad=False)
        loss = layer.forward(x).sum()
        loss.backward()

        def loss_for_weight(flat_values):
            temp_layer = Linear(2, 2)
            temp_layer.weight = Tensor(
                [[flat_values[0], flat_values[1]], [flat_values[2], flat_values[3]]],
                dtype="float32",
                requires_grad=True,
            )
            temp_layer.bias = Tensor([bias_values[0], bias_values[1]], dtype="float32", requires_grad=True)
            inputs = Tensor([x_values], dtype="float32")
            return temp_layer.forward(inputs).sum()[0]

        def loss_for_bias(flat_values):
            temp_layer = Linear(2, 2)
            temp_layer.weight = Tensor(
                [[weight_values[0], weight_values[1]], [weight_values[2], weight_values[3]]],
                dtype="float32",
                requires_grad=True,
            )
            temp_layer.bias = Tensor([flat_values[0], flat_values[1]], dtype="float32", requires_grad=True)
            inputs = Tensor([x_values], dtype="float32")
            return temp_layer.forward(inputs).sum()[0]

        self.assert_grad_close(layer.weight.grad.data(), _finite_difference(loss_for_weight, weight_values))
        self.assert_grad_close(layer.bias.grad.data(), _finite_difference(loss_for_bias, bias_values))

    def test_broadcast_add_bias_gradient_matches_finite_difference(self):
        x_values = [1.0, 2.0, 3.0, 4.0]
        bias_values = [0.25, -0.5]

        x = Tensor([[x_values[0], x_values[1]], [x_values[2], x_values[3]]], dtype="float32")
        bias = Tensor([bias_values[0], bias_values[1]], dtype="float32", requires_grad=True)
        loss = (x + bias).sum()
        loss.backward()

        def loss_for_bias(flat_values):
            tensor = Tensor([[x_values[0], x_values[1]], [x_values[2], x_values[3]]], dtype="float32")
            bias_tensor = Tensor([flat_values[0], flat_values[1]], dtype="float32", requires_grad=True)
            return (tensor + bias_tensor).sum()[0]

        self.assert_grad_close(bias.grad.data(), _finite_difference(loss_for_bias, bias_values))


if __name__ == "__main__":
    unittest.main()
