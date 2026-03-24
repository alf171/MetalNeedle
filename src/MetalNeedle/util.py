from functools import reduce
from typing import Iterable, Union

class TensorUtils:
    @staticmethod
    def flatten(array):
        return [
            item
            for sublist in array
            for item in (
                TensorUtils.flatten(sublist) if isinstance(sublist, list) else [sublist]
            )
        ]

    @staticmethod
    def get_shape(data) -> list[int]:
        if isinstance(data, list):
            shape = []
            current_level = data
            while isinstance(current_level, list):
                shape.append(len(current_level))
                current_level = current_level[0] if len(current_level) > 0 else []
            return shape
        else:
            raise ValueError("tensor data must be a list")

    @staticmethod
    def cartesian_product(array):
        if not array:
            return [[]]

        rest = TensorUtils.cartesian_product(array[1:])

        res = []
        for start_item in array[0]:
            for remaining_item in rest:
                res.append([start_item] + remaining_item)

        return res

    @staticmethod
    def product(lst: Iterable[int] | tuple[int]) -> int:
        return reduce(lambda x, y: x * y, lst, 1)

    @staticmethod
    def can_broadcast(shape1, shape2):
        """
        Check if two shapes are compatible for broadcasting
        """
        r_shape1 = list(reversed(shape1))
        r_shape2 = list(reversed(shape2))
        for i in range(min(len(r_shape1), len(r_shape2))):
            if r_shape1[i] != r_shape2[i] and r_shape1[i] != 1 and r_shape2[i] != 1:
                return False

        return True

    @staticmethod
    def normalize_axes(axes: Union[int, list[int], None], shape: list[int], operation: str):
        """
        Handle negative indexes and partial indexing
        """
        if axes is None or axes == []:
            iterable_axes = range(len(shape))
        elif isinstance(axes, int):
            iterable_axes = [axes]
        else:
            iterable_axes = axes

        normalized_axes = []
        ndim = len(shape)
        for axis in iterable_axes:
            normalized_axis = (ndim + axis) if axis < 0 else axis
            if not (0 <= normalized_axis < ndim):
                raise ValueError(f"[{operation}] index out of range")
            normalized_axes.append(normalized_axis)
        return normalized_axes
