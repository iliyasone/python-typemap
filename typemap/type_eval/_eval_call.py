import annotationlib
import enum
import inspect
import types
import typing
import typing_extensions

from typing import Any


from . import _eval_operators
from . import _eval_typing
from . import _typing_inspect
from ._eval_operators import _callable_type_to_signature
from ._apply_generic import substitute, get_annotations

RtType = Any

from typing import _UnpackGenericAlias  # type: ignore [attr-defined]  # noqa: PLC2701


def _type(t):
    if t is None or isinstance(t, (int, str, bool, bytes, enum.Enum)):
        return typing.Literal[t]
    elif isinstance(t, type):
        return type[t]
    else:
        return type(t)


def eval_call(func: types.FunctionType, /, *args: Any, **kwargs: Any) -> RtType:
    arg_types = tuple(_type(t) for t in args)
    kwarg_types = {k: _type(t) for k, t in kwargs.items()}
    return eval_call_with_types(func, *arg_types, **kwarg_types)


def _get_bound_type_args(
    func: types.FunctionType,
    arg_types: tuple[RtType, ...],
    kwarg_types: dict[str, RtType],
) -> dict[object, RtType]:
    # Run in ForwardRef mode so that if one of the arguments or if the
    # return value crashes due to a bad attribution projection or
    # something, the others will survive it.
    sig = inspect.signature(
        func, annotation_format=annotationlib.Format.FORWARDREF
    )

    bound = sig.bind(*arg_types, **kwarg_types)

    bound_type_args: dict[object, RtType] = {}
    for tv, tp in _get_bound_type_args_from_bound_args(sig, bound).items():
        bound_type_args[tv] = tp
        if name := getattr(tv, "__name__", None):
            bound_type_args[name] = tp
    return bound_type_args


def _get_bound_type_args_from_bound_args(
    sig: inspect.Signature,
    bound: inspect.BoundArguments,
) -> dict[object, RtType]:
    vars: dict[object, RtType] = {}
    _update_bound_self_from_receiver(sig, bound, vars)
    # TODO: duplication, error cases
    for param in sig.parameters.values():
        # Unpack[TypeVarType] for *args
        if (
            param.kind == inspect.Parameter.VAR_POSITIONAL
            # XXX: typing_extensions also
            and isinstance(param.annotation, _UnpackGenericAlias)
            and param.annotation.__args__
            and (tv := param.annotation.__args__[0])
            # XXX: should we allow just a regular one with a tuple bound also?
            # maybe! it would match what I want to do for kwargs!
            and isinstance(tv, typing.TypeVarTuple)
        ):
            tps = bound.arguments.get(param.name, ())
            vars[tv] = tuple[tps]  # type: ignore[valid-type]
        # Unpack[T] for **kwargs
        elif (
            param.kind == inspect.Parameter.VAR_KEYWORD
            # XXX: typing_extensions also
            and isinstance(param.annotation, _UnpackGenericAlias)
            and param.annotation.__args__
            and (tv := param.annotation.__args__[0])
            # XXX: should we allow just a regular one with a tuple bound also?
            # maybe! it would match what I want to do for kwargs!
            and isinstance(tv, typing.TypeVar)
            and tv.__bound__
            and typing_extensions.is_typeddict(tv.__bound__)
        ):
            tp = typing.TypedDict(f"**{param.name}", bound.kwargs)  # type: ignore[misc]
            vars[tv] = tp
        # trivial type[T] bindings
        elif (
            _typing_inspect.is_generic_alias(param.annotation)
            and param.annotation.__origin__ is type
            and (tv := param.annotation.__args__[0])
            and isinstance(tv, typing.TypeVar)
            and (arg := bound.arguments.get(param.name))
            and _typing_inspect.is_generic_alias(arg)
            and arg.__origin__ is type
        ):
            vars[tv] = arg.__args__[0]
        # trivial T bindings
        elif (
            _is_self_type(param.annotation)
            or isinstance(param.annotation, typing.TypeVar)
            or _typing_inspect.is_generic_alias(param.annotation)
        ):
            param_value = bound.arguments[param.name]
            _update_bound_typevar(
                param.name, param.annotation, param_value, vars
            )
        # TODO: simple bindings to other variables too

    return vars


def _update_bound_typevar(
    param_name: str,
    tv: Any,
    param_value: Any,
    vars: dict[object, RtType],
) -> None:
    if _is_self_type(tv):
        _update_bound_var(param_name, typing.Self, "Self", param_value, vars)
    elif isinstance(tv, typing.TypeVar):
        _update_bound_var(param_name, tv, tv.__name__, param_value, vars)
    elif isinstance(tv, typing.TypeVarTuple):
        if tv not in vars:
            vars[tv] = param_value
        elif vars[tv] != param_value:
            raise ValueError(
                f"Type variable {tv.__name__} "
                f"is already bound to {_type_name(vars[tv])}, "
                f"but got {_type_name(param_value)}"
            )
    elif _typing_inspect.is_generic_alias(tv):
        tv_args = tv.__args__

        with _eval_typing._ensure_context() as ctx:
            param_args = _eval_operators._get_args(
                param_value, tv.__origin__, ctx
            )

        if param_args is None:
            raise ValueError(f"Argument type mismatch for {param_name}")

        for p_arg, c_arg in zip(tv_args, param_args, strict=True):
            _update_bound_typevar(param_name, p_arg, c_arg, vars)


def _is_self_type(tp: Any) -> bool:
    return tp is typing.Self or tp is typing_extensions.Self


def _contains_self(tp: Any) -> bool:
    if _is_self_type(tp):
        return True
    if isinstance(tp, list):
        return any(_contains_self(arg) for arg in tp)
    if _typing_inspect.is_generic_alias(tp) or isinstance(tp, types.UnionType):
        return any(_contains_self(arg) for arg in typing.get_args(tp))
    return False


def _signature_contains_self(sig: inspect.Signature) -> bool:
    return _contains_self(sig.return_annotation) or any(
        _contains_self(param.annotation) for param in sig.parameters.values()
    )


def _unwrap_type_argument(tp: RtType) -> RtType:
    if (
        _typing_inspect.is_generic_alias(tp)
        and typing.get_origin(tp) is type
        and typing.get_args(tp)
    ):
        return typing.get_args(tp)[0]
    return tp


def _update_bound_self_from_receiver(
    sig: inspect.Signature,
    bound: inspect.BoundArguments,
    vars: dict[object, RtType],
) -> None:
    if not _signature_contains_self(sig):
        return

    params = tuple(sig.parameters.values())
    if not params:
        return

    first = params[0]
    if first.name not in bound.arguments:
        return

    if first.name == "self" or _is_self_type(first.annotation):
        self_type = bound.arguments[first.name]
    elif first.name == "cls" or (
        _typing_inspect.is_generic_alias(first.annotation)
        and typing.get_origin(first.annotation) is type
        and typing.get_args(first.annotation)
        and _contains_self(typing.get_args(first.annotation)[0])
    ):
        self_type = _unwrap_type_argument(bound.arguments[first.name])
    else:
        return

    _update_bound_var(first.name, typing.Self, "Self", self_type, vars)


def _update_bound_var(
    param_name: str,
    var: object,
    var_name: str,
    param_value: RtType,
    vars: dict[object, RtType],
) -> None:
    if var not in vars:
        vars[var] = param_value
    elif vars[var] != param_value:
        raise ValueError(
            f"Type variable {var_name} "
            f"is already bound to {_type_name(vars[var])}, "
            f"but got {_type_name(param_value)}"
        )


def _type_name(tp: RtType) -> str:
    return getattr(tp, "__name__", repr(tp))


def eval_call_with_types(
    func: types.FunctionType | typing.Callable[..., Any],
    *arg_types: RtType,
    **kwarg_types: RtType,
) -> RtType:
    if isinstance(func, types.FunctionType):
        vars: dict[object, Any] = _get_bound_type_args(
            func, arg_types, kwarg_types
        )
        for p in func.__type_params__:
            if p.__name__ not in vars:
                vars[p.__name__] = p

        return eval_func_with_type_vars(func, vars)

    else:
        from typemap.typing import GenericCallable

        resolved_callable = _eval_typing.eval_typing(func)

        if (
            _typing_inspect.is_generic_alias(resolved_callable)
            and resolved_callable.__origin__ is GenericCallable
        ):
            typevars_tuple, callable_lambda = typing.get_args(resolved_callable)
            type_vars = typing.get_args(typevars_tuple)
            resolved_callable = callable_lambda(*type_vars)
            # Evaluate the result to expand type aliases
            resolved_callable = _eval_typing.eval_typing(resolved_callable)

        sig = _callable_type_to_signature(resolved_callable)
        bound = sig.bind(*arg_types, **kwarg_types)
        bound_args = _get_bound_type_args_from_bound_args(sig, bound)
        res = substitute(sig.return_annotation, bound_args)

        return res


def eval_func_with_type_vars(
    func: types.FunctionType, vars: dict[object, RtType]
) -> RtType:
    with _eval_typing._ensure_context() as ctx:
        return _eval_call_with_type_vars(func, vars, ctx)


def _eval_call_with_type_vars(
    func: types.FunctionType,
    vars: dict[object, RtType],
    ctx: _eval_typing.EvalContext,
) -> RtType:
    old_obj = ctx.current_generic_alias
    ctx.current_generic_alias = func
    try:
        rr = get_annotations(func, vars)
        if rr is None:
            return Any
        ret = substitute(rr["return"], vars)
        return _eval_typing.eval_typing(ret)
    finally:
        ctx.current_generic_alias = old_obj
