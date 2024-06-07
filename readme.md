## What ##
A simple library for building runnable tree structures. Trees are built using Nodes, which
contain code to be run.

## Zen ##
1. Follow established names: a Node is a Node, not a "Leaf".
2. Syntax sugar is sweet in moderation.
3. Give the programmer granular control.

## Axioms ##
1) A tree can only have one root node.
2) Nodes can be run concurrently.
3) Dictionary outputs are passed as kwargs to children. All other types are passed as args.
4) UnionNodes can never be direct children of PickerNodes. Safety first!
5) Be careful with UnionNodes, they can cause invisible deadlocks.
6) I said to be careful with UnionNodes, they can cause invisible deadlocks!

## How to Use During Development ##
- `pip install -e .` to install on your environment
- `pytest` to run tests, add `-s` flag for tests to run `print` statements

## Extras ##
https://www.youtube.com/watch?v=9ZhbA0FHZYc&ab_channel=MatthewBerman