## What ##
A simple library for building runnable async trees. Trees are a web of interconnected Nodes, which contain code to be run. The number of workers is automatically managed (optional).

## Use

**Building a tree with the `|` operator**
```python
# Declare your nodes
root_node = Node(...)
child1 = Node(...)
child2 = Node(...)
grandchild = Node(...)

# Set the layout
nodes = {
    root_node: {
        child1: [grandchild],
        child2: [grandchild], # grandchild_node will wait for child1 and child2 to complete before running
    }
}

# Use the '|' operator to connect the nodes
executor = AsyncTreeExecutor(logger=logger)
tree = executor | nodes
result = await tree.run()
```

**Connecting nodes manually**
```python
root_node = Node(...)
child1 = Node(...)

await root_node.connect(child1)
```


**Evaluating coroutine kwargs during runtime**
```python
node = Node(
    coroutine=my_coroutine
    kwargs=dict(
        my_arg=lambda: my_arg
    )
)
```

**Altering a tree during runtime**
updated example coming later

Powered by `asyncio` (https://docs.python.org/3/library/asyncio.html)

## How ##
- You have a tree of interconected `Nodes` and an `asyncio.Queue()`
- Upon each Node's execution, it removes itself from the queue and enqueues its children up next

## Axioms ##
1) Children start running as soon as all their parent's are finished.
2) There's no passing of state between nodes - you can handle that however you see fit

## Important ##
- Node properties are generally accessible, but are immutable during a node's runtime (do not confuse with the tree's runtime).
- Coroutines and callbacks will always receive the `node` as their first (positional) argument. Everything else if a `keyword argument`.
- `on_before_run` and `on_after_run` callbacks must be asynchronous

## Installation ##
- `pip install grafo` to install on your environment
- `pytest` to run tests, add `-s` flag for tests to run `print` statements

## Zen ##
1. Follow established names: a Node is a Node, not a "Leaf".
2. Syntax sugar is sweet in moderation.
3. Give the programmer granular control.
