## What ##
A simple library for building runnable tree structures. Trees are built using Nodes, which
contain code to be run.

## Zen ##
1. Follow established names: a Node is a Node, not a "Leaf".
2. Syntax sugar is sweet in moderation.
3. Give the programmer granular control.

## Axioms ##
1) A tree can only have one root node.
2) Nodes are executed concurrently.
3) Upon finishing execution, a node attempts to queue it's children for execution.