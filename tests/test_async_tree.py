import asyncio
import logging
from typing import Any, Callable, List, Optional, Union
from uuid import uuid4

import pytest

from grafo.trees import AsyncTreeExecutor, Node, PickerNode, UnionNode
from grafo._internal import logger

logger.setLevel(logging.DEBUG)


# Auxiliary functions
def create_node(
    name: str,
    coroutine: Any,
    picker: Optional[Any] = None,
    is_union_node: bool = False,
    timeout: Optional[float] = None,
    on_after_run: Optional[Callable[..., Any]] = None,
    on_after_run_kwargs: Optional[dict[str, Any]] = None,
) -> Union[PickerNode, UnionNode, Node]:
    """
    Create a node with the given name, coroutine, and picker function.
    """
    metadata = {"name": name, "description": f"{name.capitalize()} Node"}
    if picker:
        return PickerNode(
            uuid=str(uuid4()),
            metadata=metadata,
            args=[name],
            coroutine=picker,
            # forward_output=True,
        )
    if is_union_node:
        node = UnionNode(
            uuid=str(uuid4()),
            metadata=metadata,
            coroutine=coroutine,
            args=[name],
            timeout=timeout,
            # forward_output=True,
            on_after_run=on_after_run,
            on_after_run_kwargs=on_after_run_kwargs,
        )
        return node

    return Node(
        uuid=str(uuid4()),
        metadata=metadata,
        coroutine=coroutine,
        args=[name],
        # forward_output=True,
    )


async def mockup_coroutine(name):
    """
    Example coroutine function that simulates a task that takes 1 second to complete.
    """
    await asyncio.sleep(1)
    # logger.debug(f"{name} executed")
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
    logger.debug(f"{node.metadata['name']} executed")
    logger.debug(
        f"  -> picked: {children[0].metadata['name']}, {children[2].metadata['name']}"
    )
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

    executor = AsyncTreeExecutor(name="Manual Tree", root=root_node, num_workers=3)
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
    executor = AsyncTreeExecutor(name="Picker Tree")
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
    executor = AsyncTreeExecutor(
        name="Union Tree", use_dynamic_workers=False, num_workers=10
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
    executor = AsyncTreeExecutor(cutoff_branch_on_error=True)
    tree = executor | nodes
    result = await tree.run()

    # Assert result
    nodes_uuids = [root_node.uuid, child_node1.uuid]
    assert all(node.uuid in nodes_uuids for node in result)
    assert child_node2.uuid not in nodes_uuids
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
    executor = AsyncTreeExecutor(quit_tree_on_error=True)
    tree = executor | nodes
    result = await tree.run()

    # Assert result
    nodes_uuids = [root_node.uuid, child_node1.uuid]
    assert all(node.uuid in nodes_uuids for node in result)
    assert child_node2.uuid not in nodes_uuids
    logger.debug(result)


@pytest.mark.asyncio
async def test_union_node_timeout():
    """
    Test the AsyncTreeExecutor with a UnionNode that has a timeout.
    """

    async def long_running_coroutine(name):
        # Simulate a long-running task
        await asyncio.sleep(3)
        logger.debug(f"{name} executed")
        return f"{name} result"

    root_node = create_node("root", mockup_coroutine)
    child_node1 = create_node("child1", long_running_coroutine)
    child_node2 = create_node("child2", mockup_coroutine)
    # Create a UnionNode with a timeout of 1 second
    union_node: UnionNode = create_node(
        "union", mockup_coroutine, is_union_node=True, timeout=1
    )  # type: ignore

    # Using a JSON-like structure and the '|' operator to build the tree
    nodes = {
        root_node: {
            child_node1: [union_node],
            child_node2: [union_node],
        }
    }

    # Forward the results of each node to its children as arguments
    executor = AsyncTreeExecutor(quit_tree_on_error=True)
    tree = executor | nodes
    result = await tree.run()

    # Assert that the root and child nodes completed successfully
    nodes_uuids = [root_node.uuid, child_node1.uuid]
    assert all(node.uuid in nodes_uuids for node in result)
    assert union_node.uuid not in nodes_uuids
    logger.debug(result)


@pytest.mark.asyncio
async def test_run_and_yield():
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
    executor = AsyncTreeExecutor(name="Yielding Tree")
    tree = executor | nodes
    results = []

    stop_event = asyncio.Event()
    async for node in tree.yielding([stop_event]):
        results.append((node.uuid, node))
        logger.debug(f"Yielded: {node}")
        if node.uuid == grandchild_node2.uuid:
            stop_event.set()

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
async def test_run_and_yield_with_union_node():
    """
    Test the AsyncTreeExecutor's run_and_yield method with a UnionNode to ensure it yields results from all nodes, including a UnionNode.
    """
    root_node = create_node("root", mockup_coroutine)
    child_node1 = create_node("child1", mockup_coroutine)
    # Create a UnionNode by setting is_union_node=True
    union_node = create_node("union", mockup_coroutine, is_union_node=True)
    child_node2 = create_node("child2", mockup_coroutine)
    grandchild_node1 = create_node("grandchild1", mockup_coroutine)
    grandchild_node2 = create_node("grandchild2", mockup_coroutine)

    # Building the tree using a JSON-like structure
    nodes = {
        root_node: {
            child_node1: {union_node: [grandchild_node1, grandchild_node2]},
            child_node2: [union_node],
        }
    }

    executor = AsyncTreeExecutor(name="Yielding Tree with Union")
    tree = executor | nodes
    results = []

    stop_event = asyncio.Event()
    async for node in tree.yielding([stop_event]):
        results.append((node.uuid, node))
        logger.debug(f"Yielded: {node}")
        if node.uuid == grandchild_node2.uuid:
            stop_event.set()

    expected_node_ids = [
        root_node.uuid,
        child_node1.uuid,
        union_node.uuid,
        child_node2.uuid,
        grandchild_node1.uuid,
        grandchild_node2.uuid,
    ]

    yielded_ids = [n_uuid for n_uuid, _ in results]
    assert all(node_uuid in yielded_ids for node_uuid in expected_node_ids)
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

    def stop_yielding(stop_event: asyncio.Event, **kwargs):
        stop_event.set()

    stop_event = asyncio.Event()

    root_node = create_node("root", mockup_coroutine)
    child1_node = create_node("child1", long_running_coroutine)
    child2_node = create_node("child2", mockup_coroutine)
    union_node = create_node(
        "union",
        long_running_coroutine,
        is_union_node=True,
        timeout=1,
        on_after_run=stop_yielding,
        on_after_run_kwargs={"stop_event": stop_event},
    )

    # Build the tree using a JSON-like structure
    nodes = {
        root_node: {
            child1_node: [union_node],
            child2_node: [union_node],
        }
    }

    executor = AsyncTreeExecutor(
        name="Yielding Tree with Timeout", quit_tree_on_error=True
    )
    tree = executor | nodes
    results = []

    async for node in tree.yielding([stop_event]):
        results.append((node.uuid, node))
        logger.debug(f"Yielded: {node}")
        if node.uuid == child1_node.uuid:
            stop_event.set()

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
