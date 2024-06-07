import asyncio
from logging import Logger
from typing import Any, List, Optional
from uuid import uuid4

import pytest

from grafo.trees import AsyncTreeExecutor, Node, PickerNode, UnionNode

logger = Logger("ROOT")
logger.setLevel("DEBUG")


def create_node(
    name: str,
    coroutine: Any,
    picker: Optional[Any] = None,
    is_union_node: bool = False,
) -> Node:
    """
    Create a node with the given name, coroutine, and picker function.
    """
    if picker:
        return PickerNode(
            uuid=str(uuid4()),
            name=name,
            description=f"{name.capitalize()} Node",
            coroutine=coroutine,
            args=[name],
            picker=picker,
            # forward_output=True,
        )
    if is_union_node:
        return UnionNode(
            uuid=str(uuid4()),
            name=name,
            description=f"{name.capitalize()} Node",
            coroutine=coroutine,
            args=[name],
            # forward_output=True,
        )

    return Node(
        uuid=str(uuid4()),
        name=name,
        description=f"{name.capitalize()} Node",
        coroutine=coroutine,
        args=[name],
        # forward_output=True,
    )


async def example_coroutine(name):
    """
    Example coroutine function that simulates a task that takes 1 second to complete.
    """
    await asyncio.sleep(0.5)
    print(f"{name} executed")
    return f"{name} result"


async def example_picker(node: Node, result: Any, children: List[Node]):
    """
    Example picker function that selects the first and third children of the root node.
    """
    if "root" in node.name:
        return [children[0], children[2]]
    return []


@pytest.mark.asyncio
async def test_async_tree_manual():
    """
    Test the AsyncTreeExecutor using manual connections between nodes to build the tree.
    """
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


@pytest.mark.asyncio
async def test_async_tree():
    """
    Test the AsyncTreeExecutor using a JSON-like structure to build the tree.
    """
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
    executor = AsyncTreeExecutor()
    tree = executor | nodes
    result = await tree.run()
    print(result)  # NOTE: run pytest with flag '-s' to see the print output


@pytest.mark.asyncio
async def test_async_tree_with_union_node():
    """
    Test the AsyncTreeExecutor using a JSON-like structure to build the tree with a UnionNode.
    """
    root_node = create_node(
        "root", example_coroutine
    )  # NOTE: Careful about pickers, a picker who picks between two parents of a UnionNode can loop forever
    child_node1 = create_node("child1", example_coroutine)
    child_node2 = create_node("child2", example_coroutine)
    child_node3 = create_node("child3", example_coroutine)
    grandchild_node1 = create_node("grandchild1", example_coroutine, is_union_node=True)
    grandchild_node2 = create_node("grandchild2", example_coroutine)

    # Using a JSON-like structure and the '|' operator to build the tree
    nodes = {
        root_node: {
            child_node1: None,
            child_node2: [grandchild_node1],
            child_node3: {grandchild_node1: [grandchild_node2]},
        }
    }

    # Forward the results of each node to its children as arguments
    executor = AsyncTreeExecutor(cutoff_branch_on_error=True, logger=logger)
    tree = executor | nodes
    result = await tree.run()
    print(result)  # NOTE: run pytest with flag '-s' to see the print output


asyncio.run(test_async_tree_with_union_node())
