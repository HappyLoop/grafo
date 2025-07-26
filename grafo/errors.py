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
