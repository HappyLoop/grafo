import asyncio
from typing import Any, List, Optional
from uuid import uuid4

import pytest

from grafo.trees import AsyncTreeExecutor, Node, PickerNode, UnionNode
from grafo._internal import logger


# Auxiliary functions
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
            args=[name],
            coroutine=picker,
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


async def mockup_coroutine(name):
    """
    Example coroutine function that simulates a task that takes 1 second to complete.
    """
    await asyncio.sleep(1)
    logger.debug(f"{name} executed")
    return f"{name} result"


async def mockup_bad_coroutine(name):
    """
    Example coroutine function that simulates an error.
    """
    raise ValueError(f"{name} failed!")


async def mockup_picker(node: Node, children: List[Node], *args) -> List[Node]:
    """
    Example picker function that selects the first and third children of the root node.
    """
    logger.debug(f"{node.name} executed")
    logger.debug(f"  -> picked: {children[0].name}, {children[2].name}")
    return [children[0], children[2]]


# Tests
@pytest.mark.asyncio
async def test_manual_tree():
    """
    Test the AsyncTreeExecutor using manual connections between nodes to build the tree.
    """
    root_node = create_node("root", mockup_coroutine)
    child_node1 = create_node("child1", mockup_coroutine)
    grandchild_node1 = create_node("grandchild1", mockup_coroutine)
    grandchild_node2 = create_node("grandchild2", mockup_coroutine)

    # Manually connecting nodes
    root_node.connect(child_node1)
    child_node1.connect(grandchild_node1)
    child_node1.connect(grandchild_node2)

    executor = AsyncTreeExecutor(root=root_node, num_workers=3, logger=logger)
    result = await executor.run()

    # Assert result
    nodes_uuids = [
        root_node.uuid,
        child_node1.uuid,
        grandchild_node1.uuid,
        grandchild_node2.uuid,
    ]
    assert all(node_uuid in result.keys() for node_uuid in nodes_uuids)
    logger.debug(result)


@pytest.mark.asyncio
async def test_with_picker_node():
    """
    Test the AsyncTreeExecutor using a JSON-like structure to build the tree.
    """
    root_node = create_node("root", None, mockup_picker)
    child_node1 = create_node("child1", mockup_coroutine)
    child_node2 = create_node("child2", mockup_coroutine)
    child_node3 = create_node("child3", mockup_coroutine)
    grandchild_node1 = create_node("grandchild1", mockup_coroutine)
    grandchild_node2 = create_node("grandchild2", mockup_coroutine)

    # Using a JSON-like structure and the '|' operator to build the tree
    nodes = {
        root_node: {
            child_node1: None,
            child_node2: [grandchild_node1],
            child_node3: [grandchild_node2],
        }
    }

    # Forward the results of each node to its children as arguments
    executor = AsyncTreeExecutor(logger=logger)
    tree = executor | nodes
    result = await tree.run()

    # Assert result
    nodes_uuids = [
        root_node.uuid,
        child_node1.uuid,
        child_node3.uuid,
        grandchild_node2.uuid,
    ]
    assert all(node_uuid in result.keys() for node_uuid in nodes_uuids)
    logger.debug(result)


@pytest.mark.asyncio
async def test_with_union_node():
    """
    Test the AsyncTreeExecutor using a JSON-like structure to build the tree with a UnionNode.
    """
    root_node = create_node("root", mockup_coroutine)
    child_node1 = create_node("child1", mockup_coroutine)
    child_node2 = create_node("child2", mockup_coroutine)
    child_node3 = create_node("child3", mockup_coroutine)
    grandchild_node1 = create_node("grandchild1", mockup_coroutine, is_union_node=True)
    grandchild_node2 = create_node("grandchild2", mockup_coroutine)

    # Using a JSON-like structure and the '|' operator to build the tree
    nodes = {
        root_node: {
            child_node1: None,
            child_node2: [grandchild_node1],
            child_node3: {grandchild_node1: [grandchild_node2]},
        }
    }

    # Forward the results of each node to its children as arguments
    executor = AsyncTreeExecutor(logger=logger)
    tree = executor | nodes
    result = await tree.run()

    # Assert result
    nodes_uuids = [
        root_node.uuid,
        child_node1.uuid,
        child_node2.uuid,
        child_node3.uuid,
        grandchild_node1.uuid,
        grandchild_node2.uuid,
    ]
    assert all(node_uuid in result.keys() for node_uuid in nodes_uuids)
    logger.debug(result)


@pytest.mark.asyncio
async def test_error_cutoff_branch():
    """
    Test the AsyncTreeExecutor using a JSON-like structure to build the tree with a UnionNode.
    """
    root_node = create_node("root", mockup_coroutine)
    child_node1 = create_node("child1", mockup_coroutine)
    child_node2 = create_node("child2", mockup_bad_coroutine)
    grandchild_node1 = create_node("grandchild1", mockup_bad_coroutine)

    # Using a JSON-like structure and the '|' operator to build the tree
    nodes = {
        root_node: {
            child_node1: None,
            child_node2: [grandchild_node1],
        }
    }

    # Forward the results of each node to its children as arguments
    executor = AsyncTreeExecutor(cutoff_branch_on_error=True, logger=logger)
    tree = executor | nodes
    result = await tree.run()

    # Assert result
    nodes_uuids = [root_node.uuid, child_node1.uuid]
    assert all(node_uuid in result.keys() for node_uuid in nodes_uuids)
    logger.debug(result)


@pytest.mark.asyncio
async def test_error_quit_tree():
    """
    Test the AsyncTreeExecutor using a JSON-like structure to build the tree with a UnionNode.
    """
    root_node = create_node("root", mockup_coroutine)
    child_node1 = create_node("child1", mockup_coroutine)
    child_node2 = create_node("child2", mockup_bad_coroutine)
    grandchild_node1 = create_node("grandchild1", mockup_coroutine)
    grandchild_node2 = create_node("grandchild2", mockup_coroutine)

    # Using a JSON-like structure and the '|' operator to build the tree
    nodes = {
        root_node: {
            child_node1: [grandchild_node1],
            child_node2: [grandchild_node2],
        }
    }

    # Forward the results of each node to its children as arguments
    executor = AsyncTreeExecutor(quit_tree_on_error=True, logger=logger)
    tree = executor | nodes
    result = await tree.run()

    # Assert result
    nodes_uuids = [root_node.uuid, child_node1.uuid]
    assert all(node_uuid in result.keys() for node_uuid in nodes_uuids)
    logger.debug(result)


asyncio.run(test_with_picker_node())
