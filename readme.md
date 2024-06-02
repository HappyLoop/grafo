# WHAT IS IT #
A simple library for building runnable tree structures. Trees are built using Nodes, which
contain code to be run.

# ZEN #
1) No fancy names: favor established naming conventions (i.e. a Node is a Node, not a 'Leaf').
2) Low sugar level: syntax sugar is cool, until there's layers of it. Keep it moderate.
3) Simplicity: fewer object types makes everything easier to understand.
4) Granular control: the programmer should have as much control as possible.

# AXIOMS #
1) A tree can only have one root node.
2) Nodes are executed concurrently.
3) Execution of a node begins as soon as it's parent has finished running.