import unittest

from MetalNeedle import Tensor
from MetalNeedle.loss.cat_cross_entropy import CategoricalCrossEntropy
from MetalNeedle.nn.activations.softmax import Softmax


def _loss_for_logits(logit_values, target_values):
    logits = Tensor([logit_values], dtype="float32", requires_grad=False)
    targets = Tensor([target_values], dtype="float32", requires_grad=False)
    probs = Softmax().forward(logits)
    loss = CategoricalCrossEntropy()(probs, targets)
    return loss[0]


class TestLossGrad(unittest.TestCase):
    def test_softmax_cross_entropy_gradient_matches_finite_difference(self):
        epsilon = 1e-3
        logits = Tensor([[1.2, -0.7, 0.3]], dtype="float32", requires_grad=True)
        targets = Tensor([[1.0, 0.0, 0.0]], dtype="float32", requires_grad=False)

        probs = Softmax().forward(logits)
        loss = CategoricalCrossEntropy()(probs, targets)
        loss.backward()

        analytic = logits.grad.data()
        base = logits.data()
        target_data = targets.data()
        numerical = []

        for i in range(len(base)):
            plus = base[:]
            minus = base[:]
            plus[i] += epsilon
            minus[i] -= epsilon
            loss_plus = _loss_for_logits(plus, target_data)
            loss_minus = _loss_for_logits(minus, target_data)
            numerical.append((loss_plus - loss_minus) / (2 * epsilon))

        for analytic_grad, numerical_grad in zip(analytic, numerical):
            self.assertAlmostEqual(analytic_grad, numerical_grad, places=2)


if __name__ == "__main__":
    unittest.main()
