import asyncio
import logging
from typing import Any, Callable, Optional

import pytest

from grafo.trees import AsyncTreeExecutor, Node
from grafo._internal import logger

logger.setLevel(logging.DEBUG)


# Auxiliary functions
def create_node(
    name: str,
    coroutine: Any,
    timeout: Optional[float] = None,
    on_after_run: Optional[Callable[..., Any]] = None,
    on_after_run_kwargs: Optional[dict[str, Any]] = None,
) -> Node:
    """
    Create a node with the given name, coroutine, and picker function.
    """
    return Node(
        uuid=name,
        coroutine=coroutine,
        timeout=timeout,
        on_after_run=(on_after_run, on_after_run_kwargs) if on_after_run else None,
    )


async def mockup_coroutine(node: Node):
    """
    Example coroutine function that simulates a task that takes 1 second to complete.
    """
    await asyncio.sleep(1)
    return f"{node.uuid} result"


async def mockup_picker(node: Node):
    """
    Example picker function that selects the first and third children of the root node.
    """
    logger.debug(f" -> picked: {node.children[0].uuid}, {node.children[2].uuid}")
    await node.disconnect(node.children[1])


async def mockup_bad_coroutine(_):
    """
    Example coroutine function that simulates an error.
    """
    raise ValueError("bad coroutine")


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
    await root_node.connect(child_node1)
    await child_node1.connect(grandchild_node1)
    await child_node1.connect(grandchild_node2)

    executor = AsyncTreeExecutor(uuid="Manual Tree", root=root_node, num_workers=3)
    result = await executor.run()

    # Assert result
    nodes_uuids = [
        root_node.uuid,
        child_node1.uuid,
        grandchild_node1.uuid,
        grandchild_node2.uuid,
    ]
    assert all(node.uuid in nodes_uuids for node in result)
    logger.debug(result)


@pytest.mark.asyncio
async def test_picker():
    """
    Test the AsyncTreeExecutor using a JSON-like structure to build the tree.
    """
    root_node = create_node("root", mockup_picker)
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
    executor = AsyncTreeExecutor(
        uuid="Picker Tree", use_dynamic_workers=False, num_workers=10
    )
    tree = executor | nodes
    result = await tree.run()

    # Assert result
    nodes_uuids = [
        root_node.uuid,
        child_node1.uuid,
        child_node3.uuid,
        grandchild_node2.uuid,
    ]
    assert all(node.uuid in nodes_uuids for node in result)
    assert child_node2.uuid not in nodes_uuids
    logger.debug(result)


@pytest.mark.asyncio
async def test_union():
    """
    Test the AsyncTreeExecutor using a JSON-like structure to build the tree with a UnionNode.
    """
    root_node = create_node("root", mockup_coroutine)
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
            child_node3: {grandchild_node1: [grandchild_node2]},
        }
    }

    # Forward the results of each node to its children as arguments
    executor = AsyncTreeExecutor(
        uuid="Union Tree", use_dynamic_workers=False, num_workers=10
    )
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
    assert all(node.uuid in nodes_uuids for node in result)
    logger.debug(result)


@pytest.mark.asyncio
async def test_error():
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
    executor = AsyncTreeExecutor(
        uuid="Error Tree", use_dynamic_workers=False, num_workers=10
    )
    tree = executor | nodes
    result = await tree.run()

    # Assert result
    nodes_uuids = [root_node.uuid, child_node1.uuid]
    assert all(node.uuid in nodes_uuids for node in result)
    assert child_node2.uuid not in nodes_uuids
    logger.debug(result)


@pytest.mark.asyncio
async def test_yielding():
    """
    Test the AsyncTreeExecutor's run_and_yield method to ensure it yields results as they are set.
    """
    root_node = create_node("root", mockup_coroutine)
    child_node1 = create_node("child1", mockup_coroutine)
    grandchild_node1 = create_node("grandchild1", mockup_coroutine)
    grandchild_node2 = create_node("grandchild2", mockup_coroutine)

    # Manually connecting nodes
    nodes = {
        root_node: {
            child_node1: [grandchild_node1, grandchild_node2],
        }
    }
    executor = AsyncTreeExecutor(
        uuid="Yielding Tree", use_dynamic_workers=False, num_workers=10
    )
    tree = executor | nodes
    results = []

    async for node in tree.yielding():
        results.append((node.uuid, node))
        logger.debug(f"Yielded: {node}")

    # Assert that all nodes have been processed and yielded
    nodes_uuids = [
        root_node.uuid,
        child_node1.uuid,
        grandchild_node1.uuid,
        grandchild_node2.uuid,
    ]
    yielded_uuids = [uuid for uuid, _ in results]
    assert all(node_uuid in yielded_uuids for node_uuid in nodes_uuids)
    logger.debug("All nodes yielded successfully.")


@pytest.mark.asyncio
async def test_yield_with_timeout():
    """
    Test the AsyncTreeExecutor's yielding method with a UnionNode that times out,
    ensuring that nodes that exceed the timeout do not yield a result.
    """

    async def long_running_coroutine(name):
        # Simulate a long-running task
        await asyncio.sleep(3)
        logger.debug(f"{name} executed")
        return f"{name} result"

    executor = AsyncTreeExecutor(
        uuid="Yielding Tree with Timeout", use_dynamic_workers=False, num_workers=10
    )

    root_node = create_node("root", mockup_coroutine)
    child1_node = create_node("child1", long_running_coroutine)
    child2_node = create_node("child2", mockup_coroutine)
    union_node = create_node(
        "union",
        long_running_coroutine,
        timeout=1,
    )

    # Build the tree using a JSON-like structure
    nodes = {
        root_node: {
            child1_node: [union_node],
            child2_node: [union_node],
        }
    }

    tree = executor | nodes
    results = []

    async for node in tree.yielding():
        results.append((node.uuid, node))
        logger.debug(f"Yielded: {node}")

    expected_node_ids = [
        root_node.uuid,
        child1_node.uuid,
        child2_node.uuid,
    ]

    yielded_ids = [n_uuid for n_uuid, _ in results]
    assert all(node_uuid in yielded_ids for node_uuid in expected_node_ids)
    assert union_node.uuid not in yielded_ids
    logger.debug(
        "Test yield with timeout: timed out union node did not yield result, others yielded successfully."
    )
