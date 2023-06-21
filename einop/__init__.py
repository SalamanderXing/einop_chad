__version__ = "0.0.1"
__all__ = ["einop"]

import functools
from typing import TypeVar

import einops
from einops.parsing import EinopsError, ParsedExpression

A = TypeVar("A")


@functools.lru_cache(256)
def _match_einop(pattern: str, reduction=None, **axes_lengths: int):
    """Find the corresponding operation matching the pattern"""
    split = pattern.split("->")
    if len(split) > 1:
        left, rght = pattern.split("->")

        if "," not in left:
            left = ParsedExpression(left)
            rght = ParsedExpression(rght)

            default_op = "rearrange"
            op = default_op

            for index in left.identifiers:
                if index not in rght.identifiers:
                    op = "reduce"
                    break

            for index in rght.identifiers:
                if index not in left.identifiers:
                    if op != default_op:
                        raise EinopsError(
                            "You must perform a reduce and repeat separately: {}".format(
                                pattern
                            )
                        )
                    op = "repeat"
                    break
        else:
            op = "einsum"
    else:
        op = "pack"

    return op


def einop(tensor: A, *args, reduction=None, **axes_lengths: int) -> A:
    pattern = args[-1]
    assert isinstance(pattern, str)

    """Perform either reduce, rearrange, or repeat depending on pattern"""
    op = _match_einop(pattern, reduction, **axes_lengths)
    tensor = tensor if not isinstance(tensor, tuple) else list(tensor)  # type: ignore

    if op == "rearrange":
        if reduction is not None:
            raise EinopsError(
                'Got reduction operation but there is no dimension to reduce in pattern: "{}"'.format(
                    pattern
                )
            )
        assert tensor is not None
        return einops.rearrange(tensor, pattern, **axes_lengths)
    elif op == "reduce":
        if reduction is None:
            raise EinopsError(
                "Missing reduction operation for reduce pattern: {}".format(pattern)
            )

        assert tensor is not None
        return einops.reduce(tensor, pattern, reduction, **axes_lengths)
    elif op == "repeat":
        if reduction is not None:
            raise EinopsError(
                "Do not pass reduction for repeat pattern: {}".format(pattern)
            )
        assert tensor is not None
        return einops.repeat(tensor, pattern, **axes_lengths)
    elif op == "einsum":
        if reduction is not None:
            raise EinopsError(
                "Do not pass reduction for repeat pattern: {}".format(pattern)
            )
        if len(axes_lengths) > 0:
            raise EinopsError(
                "Do not pass axis lengths for einsum pattern: {}".format(pattern)
            )
        tensors = (tensor,) + args[:-1]
        return einops.einsum(*tensors, pattern)
    elif op == "pack":
        if reduction is not None:
            raise EinopsError(
                "Do not pass reduction for repeat pattern: {}".format(pattern)
            )
        if len(axes_lengths) > 0:
            raise EinopsError(
                "Do not pass axis lengths for einsum pattern: {}".format(pattern)
            )
        return einops.pack(tensor, *args)
    else:
        raise ValueError(f"Unknown operation: {op}")
