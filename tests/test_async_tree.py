from typing import List, Optional, Any
import asyncio
from uuid import uuid4
import pytest


from grafo.trees import Node, AsyncTreeExecutor


# Helper function to create a node
def create_node(
    name: str,
    coroutine: Any,
    picker: Optional[Any] = None,
    children: Optional[List[Node]] = None,
) -> Node:
    node = Node(
        uuid=str(uuid4()),
        name=name,
        description=f"{name.capitalize()} Node",
        coroutine=coroutine,
        args=[name],
        picker=picker,
    )
    if children:
        for child in children:
            node.connect(child)
    return node


# Example coroutine function for Node
async def example_coroutine(name):
    await asyncio.sleep(1)
    print(f"Node {name} finished execution")
    return f"{name} result"


async def example_picker(node: Node, result: Any, children: List[Node]):
    if "root" in node.name:
        return [children[0], children[2]]
    return []


# Test using manual connections
@pytest.mark.asyncio
async def test_async_tree_manual():
    root_node = create_node("root", example_coroutine, example_picker)
    child_node1 = create_node("child1", example_coroutine)
    child_node2 = create_node("child2", example_coroutine)
    child_node3 = create_node("child3", example_coroutine)
    grandchild_node1 = create_node("grandchild1", example_coroutine)
    grandchild_node2 = create_node("grandchild2", example_coroutine)

    # Manually connecting nodes
    root_node.connect(child_node1)
    root_node.connect(child_node2)
    root_node.connect(child_node3)
    child_node2.connect(grandchild_node1)
    child_node3.connect(grandchild_node2)

    executor = AsyncTreeExecutor(root=root_node, num_workers=3)
    result = await executor.run()
    print(result)  # NOTE: run pytest with flag '-s' to see the print output


# Test using a JSON-like structure
@pytest.mark.asyncio
async def test_async_tree():
    root_node = create_node("root", example_coroutine, example_picker)
    child_node1 = create_node("child1", example_coroutine)
    child_node2 = create_node("child2", example_coroutine)
    child_node3 = create_node("child3", example_coroutine)
    grandchild_node1 = create_node("grandchild1", example_coroutine)
    grandchild_node2 = create_node("grandchild2", example_coroutine)

    # Using a JSON-like structure and the '|' operator to build the tree
    nodes = {
        root_node: {
            child_node1: None,
            child_node2: [grandchild_node1],
            child_node3: [grandchild_node2],
        }
    }

    # Forward the results of each node to its children as arguments
    executor = AsyncTreeExecutor(forward_results=True)
    tree = executor | nodes
    result = await tree.run()
    print(result)  # NOTE: run pytest with flag '-s' to see the print output
