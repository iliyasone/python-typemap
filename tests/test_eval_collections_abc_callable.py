# SKIP MYPY

from __future__ import annotations

import collections.abc
import typing
from typing import Callable, ParamSpec

from typemap.type_eval import eval_typing
from typemap_extensions import Params


def test_eval_collections_abc_callable_matches_typing_callable():
    """`collections.abc.Callable[...]` evaluates like `typing.Callable[...]`.

    Regression test for a bug where `eval_typing` could evaluate
    `typing.Callable[...]` but raised a ``TypeError`` for the equivalent
    `collections.abc.Callable[...]` spelling. The two spellings produce
    different alias classes (the ``collections.abc`` one flattens its
    ``__args__``), but ``typing.get_args`` normalizes both, so they must
    evaluate identically.
    """
    cases = [
        # (empty params, single param, multiple params)
        collections.abc.Callable[[], None],
        collections.abc.Callable[[int], str],
        collections.abc.Callable[[int, str], bool],
    ]
    for cabc_callable in cases:
        params, ret = typing.get_args(cabc_callable)
        typing_callable = Callable[list(params), ret]
        assert eval_typing(cabc_callable) == eval_typing(typing_callable)


def test_eval_collections_abc_callable_empty_params():
    assert eval_typing(collections.abc.Callable[[], None]) == eval_typing(
        Callable[[], None]
    )


def test_eval_collections_abc_callable_single_param():
    assert eval_typing(collections.abc.Callable[[int], str]) == eval_typing(
        Callable[[int], str]
    )


def test_eval_collections_abc_callable_multiple_params():
    assert eval_typing(
        collections.abc.Callable[[int, str], bool]
    ) == eval_typing(Callable[[int, str], bool])


def test_eval_collections_abc_callable_paramspec_passthrough():
    """A ParamSpec is passed through unchanged for both spellings."""
    P = ParamSpec("P")
    assert eval_typing(collections.abc.Callable[P, None]) == eval_typing(
        Callable[P, None]
    )


def test_eval_collections_abc_callable_params_passthrough():
    """`Params[...]` passes through for the collections.abc spelling."""
    assert eval_typing(
        collections.abc.Callable[Params[int, str], bool]
    ) == eval_typing(Callable[Params[int, str], bool])
