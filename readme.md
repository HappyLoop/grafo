## What ##
A simple library for building runnable tree structures. Trees are built using Nodes, which
contain code to be run. The number of workers is automatically managed (optional).

Use:

```
# Declare your nodes
root_node = Node(...)
child_node1 = Node(...)
child_node2 = Node(...)
grandchild_node1 = Node(...)

# Set the layout
nodes = {
    root_node: {
        child_node1: None,
        child_node2: [grandchild_node1],
    }
}

# Use the '|' operator to connect the nodes
executor = AsyncTreeExecutor(logger=logger)
tree = executor | nodes
result = await tree.run()
```

Powered by `asyncio` (https://docs.python.org/3/library/asyncio.html)

## How ##
- You have a tree of interconected `Nodes` and an `asyncio.Queue()`
- Upon each Node's execution, it removes itself from the queue and enqueues its children up next
- ⚠️ Be careful with UnionNodes, they can cause invisible deadlocks. ⚠️

## Axioms ##
1) A tree can only have one root node.
2) Nodes can be run concurrently.
3) Dictionary outputs are passed as kwargs to children. All other types are passed as args.
4) UnionNodes can never be direct children of PickerNodes.

## Zen ##
1. Follow established names: a Node is a Node, not a "Leaf".
2. Syntax sugar is sweet in moderation.
3. Give the programmer granular control.

## During Development ##
- `pip install grafo` to install on your environment
- `pytest` to run tests, add `-s` flag for tests to run `print` statements
