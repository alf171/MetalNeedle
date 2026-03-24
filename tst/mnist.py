import cProfile
import os
import pstats
from dataclasses import dataclass, field
from time import perf_counter

from MetalNeedle.loader import MnistDataLoader
from MetalNeedle.loss.cat_cross_entropy import CategoricalCrossEntropy
from MetalNeedle.nn import Linear, ReLU, Softmax
from MetalNeedle.optim.sgd import SGD


@dataclass
class PhaseTimer:
    total: float = 0.0
    count: int = 0
    min_time: float = float("inf")
    max_time: float = 0.0

    def add(self, elapsed: float) -> None:
        self.total += elapsed
        self.count += 1
        if elapsed < self.min_time:
            self.min_time = elapsed
        if elapsed > self.max_time:
            self.max_time = elapsed

    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count

    def min(self) -> float:
        if self.count == 0:
            return 0.0
        return self.min_time


@dataclass
class BatchMetrics:
    data_wait: float = 0.0
    forward: float = 0.0
    zero_grad: float = 0.0
    backward: float = 0.0
    optimizer_step: float = 0.0
    accuracy: float = 0.0
    total: float = 0.0


@dataclass
class BenchmarkMetrics:
    phases: dict[str, PhaseTimer] = field(
        default_factory=lambda: {
            "data_wait": PhaseTimer(),
            "forward": PhaseTimer(),
            "zero_grad": PhaseTimer(),
            "backward": PhaseTimer(),
            "optimizer_step": PhaseTimer(),
            "accuracy": PhaseTimer(),
            "batch_total": PhaseTimer(),
        }
    )

    def add_batch(self, batch: BatchMetrics) -> None:
        self.phases["data_wait"].add(batch.data_wait)
        self.phases["forward"].add(batch.forward)
        self.phases["zero_grad"].add(batch.zero_grad)
        self.phases["backward"].add(batch.backward)
        self.phases["optimizer_step"].add(batch.optimizer_step)
        self.phases["accuracy"].add(batch.accuracy)
        self.phases["batch_total"].add(batch.total)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _format_ms(value: float) -> str:
    return f"{value * 1000:.3f}ms"


def _print_phase_summary(metrics: BenchmarkMetrics) -> None:
    total_batch_time = metrics.phases["batch_total"].total
    print("[mnist][timing] aggregate phase summary")
    for phase_name in (
        "data_wait",
        "forward",
        "zero_grad",
        "backward",
        "optimizer_step",
        "accuracy",
        "batch_total",
    ):
        phase = metrics.phases[phase_name]
        percent = 0.0
        if total_batch_time > 0 and phase_name != "batch_total":
            percent = (phase.total / total_batch_time) * 100
        if phase_name == "batch_total":
            print(
                f"[mnist][timing] {phase_name:>14} total={_format_ms(phase.total)} "
                f"avg={_format_ms(phase.avg())} min={_format_ms(phase.min())} "
                f"max={_format_ms(phase.max_time)} count={phase.count}"
            )
            continue
        print(
            f"[mnist][timing] {phase_name:>14} total={_format_ms(phase.total)} "
            f"avg={_format_ms(phase.avg())} min={_format_ms(phase.min())} "
            f"max={_format_ms(phase.max_time)} pct={percent:6.2f}"
        )


def _print_profile(stats: pstats.Stats, lines: int) -> None:
    print("[mnist][profile] cumulative")
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(lines)
    print("[mnist][profile] internal time")
    stats.sort_stats(pstats.SortKey.TIME).print_stats(lines)
    print("[mnist][profile] callers")
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_callers(lines)


def _argmax(values, start, width):
    best_idx = 0
    best_val = values[start]
    for i in range(1, width):
        current = values[start + i]
        if current > best_val:
            best_val = current
            best_idx = i
    return best_idx


def _batch_accuracy(predictions, labels) -> float:
    pred_data = predictions.data()
    label_data = labels.data()
    num_classes = predictions.shape()[1]
    batch_size = predictions.shape()[0]
    correct = 0

    for row in range(batch_size):
        offset = row * num_classes
        pred_class = _argmax(pred_data, offset, num_classes)
        true_class = _argmax(label_data, offset, num_classes)
        if pred_class == true_class:
            correct += 1

    return correct / batch_size


def _mean_abs_tensor_value(tensor) -> float:
    values = tensor.data()
    return sum(abs(v) for v in values) / max(len(values), 1)


def _parameter_update_mean_abs(before, tensor) -> float:
    after = tensor.data()
    return sum(abs(a - b) for a, b in zip(after, before)) / max(len(before), 1)


# Input (784) -> Linear (784->128) -> ReLU -> Linear (128->10) -> Softmax
def mnist() -> None:
    device = os.getenv("MNIST_DEVICE", "cpu")
    batch_size = int(os.getenv("MNIST_BATCH_SIZE", "64"))
    epochs = int(os.getenv("MNIST_EPOCHS", "1"))
    max_batches = int(os.getenv("MNIST_MAX_BATCHES", "50"))
    reuse_first_batch = os.getenv("MNIST_REUSE_FIRST_BATCH", "0") == "1"
    per_batch_timing = _env_flag("MNIST_TIMING_PER_BATCH")
    timing_summary = _env_flag("MNIST_TIMING_SUMMARY", True)
    capture_profile = _env_flag("MNIST_ENABLE_CPROFILE", True)
    profile_lines = int(os.getenv("MNIST_PROFILE_LINES", "20"))

    data = MnistDataLoader(
        "data/mnist/train-images",
        "data/mnist/train-labels",
        batch_size,
        device=device,
    )
    layer1 = Linear(784, 128, device=device)
    relu = ReLU()
    layer2 = Linear(128, 10, device=device)
    softmax = Softmax()
    loss_fn = CategoricalCrossEntropy()

    # optim step
    parameters = [layer1.weight, layer1.bias, layer2.weight, layer2.bias]
    optimizer = SGD(parameters, lr=0.1, momentum=0.0)
    metrics = BenchmarkMetrics()

    for epoch in range(epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        batch_count = 0
        first_batch = None
        batch_fetch_start = perf_counter()
        for batch_idx, batch in enumerate(data.iter_batches(max_batches=max_batches)):
            batch_start = perf_counter()
            batch_metrics = BatchMetrics(data_wait=batch_start - batch_fetch_start)
            if reuse_first_batch:
                if first_batch is None:
                    first_batch = batch
                images, labels = first_batch
            else:
                images, labels = batch

            forward_start = perf_counter()
            l1 = layer1.forward(images)
            l2 = relu.forward(l1)
            l3 = layer2.forward(l2)
            output = softmax.forward(l3)
            loss_value = loss_fn(output, labels)
            batch_metrics.forward = perf_counter() - forward_start

            zero_grad_start = perf_counter()
            optimizer.zero_grad()
            batch_metrics.zero_grad = perf_counter() - zero_grad_start

            backward_start = perf_counter()
            loss_value.backward()
            batch_metrics.backward = perf_counter() - backward_start

            step_start = perf_counter()
            optimizer.step()
            batch_metrics.optimizer_step = perf_counter() - step_start

            accuracy_start = perf_counter()
            running_loss += float(loss_value.tensor_data.get_single_item([0]))
            running_accuracy += _batch_accuracy(output, labels)
            batch_metrics.accuracy = perf_counter() - accuracy_start
            batch_count += 1
            batch_metrics.total = perf_counter() - batch_start
            metrics.add_batch(batch_metrics)
            print(
                f"[mnist] device={device} epoch={epoch} batch={batch_idx} "
                f"loss={loss_value[0]:.6f} acc={running_accuracy / batch_count:.3f}"
            )
            if per_batch_timing:
                print(
                    f"[mnist][timing] epoch={epoch} batch={batch_idx} "
                    f"data_wait={_format_ms(batch_metrics.data_wait)} "
                    f"forward={_format_ms(batch_metrics.forward)} "
                    f"zero_grad={_format_ms(batch_metrics.zero_grad)} "
                    f"backward={_format_ms(batch_metrics.backward)} "
                    f"step={_format_ms(batch_metrics.optimizer_step)} "
                    f"accuracy={_format_ms(batch_metrics.accuracy)} "
                    f"total={_format_ms(batch_metrics.total)}"
                )
            batch_fetch_start = perf_counter()

        avg_loss = running_loss / max(batch_count, 1)
        avg_accuracy = running_accuracy / max(batch_count, 1)
        print(
            f"[mnist] device={device} epoch={epoch} "
            f"avg_loss={avg_loss:.6f} avg_acc={avg_accuracy:.3f}"
        )

    if timing_summary:
        _print_phase_summary(metrics)


if __name__ == "__main__":
    capture_profile = _env_flag("MNIST_ENABLE_CPROFILE", True)
    profile_lines = int(os.getenv("MNIST_PROFILE_LINES", "20"))
    if capture_profile:
        profiler = cProfile.Profile()
        profiler.enable()
        mnist()
        profiler.disable()
        _print_profile(pstats.Stats(profiler), profile_lines)
    else:
        mnist()
