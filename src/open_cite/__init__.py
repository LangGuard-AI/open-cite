__all__ = ["OpenCiteClient"]


def __getattr__(name):
    if name == "OpenCiteClient":
        from .client import OpenCiteClient
        return OpenCiteClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
