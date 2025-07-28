class SafeExecutionError(Exception):
    """
    Exception raised when a method is called on a running node.
    """


class NotAsyncCallableError(Exception):
    """
    Exception raised when a method is called on a node that is not an async callable.
    """


class ForwardingOverrideError(Exception):
    """
    Exception raised when a node attempts to forward its output to a child that already has an argument with the same name.
    """


class MismatchChunkType(Exception):
    """
    Exception raised when a node's yield is a chunk of a different type than the node's output type.
    """
