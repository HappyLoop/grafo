## What ##
A simple library for building runnable tree structures. Trees are built using Nodes, which
contain code to be run.

Based on `asyncio` & `instructor`!

- `asyncio`: https://docs.python.org/3/library/asyncio.html
- `instructor`: https://python.useinstructor.com/

[![Pydantic is All You Need]](https://www.youtube.com/embed/yj-wSRJwrrc)

## HOW ##
- We have a tree of interconected `Nodes` and an `asyncio.Queue()`
- Upon each Node's execution, it queues its children up next
- Workers stop when they find a `None` in the queue
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
- `pip install -e .` to install on your environment
- `pytest` to run tests, add `-s` flag for tests to run `print` statements

## Extras ##
https://www.youtube.com/watch?v=9ZhbA0FHZYc&ab_channel=MatthewBerman
