import asyncio
from grafo.trees.components import Chunk
import logging
import random
from typing import Any, Callable, Optional

import pytest

from grafo._internal import logger
from grafo.trees import AsyncTreeExecutor, Node

logger.setLevel(logging.INFO)


# Auxiliary functions
def create_node(
    name: str,
    coroutine: Any,
    timeout: Optional[float] = None,
    on_after_run: Optional[Callable[..., Any]] = None,
    on_after_run_kwargs: Optional[dict[str, Any]] = None,
    kwargs: Optional[dict[str, Any]] = None,
) -> Node:
    """
    Create a node with the given name, coroutine, and picker function.
    """
    node = Node(
        uuid=name,
        coroutine=coroutine,
        timeout=timeout,
        on_after_run=(on_after_run, on_after_run_kwargs) if on_after_run else None,
    )
    node.kwargs = dict(node=node)
    node.kwargs.update(kwargs or {})
    return node


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
    logger.info(f" -> picked: {node.children[0].uuid}, {node.children[2].uuid}")
    await node.disconnect(node.children[1])


async def mockup_bad_coroutine(node: Node):
    """
    Example coroutine function that simulates an error.
    """
    raise ValueError(f"{node.uuid} bad coroutine")


async def cycle_coroutine(node: Node, child_node: Node):
    """
    Example coroutine function that simulates a cycle.
    """
    logger.info(f"Cycle coroutine: {node.uuid} -> {child_node.uuid}")
    await node.connect(child_node)
    for grandchild in child_node.children:
        await child_node.disconnect(grandchild)


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

    executor = AsyncTreeExecutor(uuid="Manual Tree", roots=[root_node], num_workers=3)
    result = await executor.run()

    # Assert result
    nodes_uuids = [
        root_node.uuid,
        child_node1.uuid,
        grandchild_node1.uuid,
        grandchild_node2.uuid,
    ]
    assert all(node.uuid in nodes_uuids for node in result)
    logger.info(result)


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

    await root_node.connect(child_node1)
    await root_node.connect(child_node2)
    await root_node.connect(child_node3)
    await child_node2.connect(grandchild_node1)
    await child_node3.connect(grandchild_node2)

    executor = AsyncTreeExecutor(
        uuid="Picker Tree", roots=[root_node], use_dynamic_workers=False, num_workers=10
    )
    result = await executor.run()

    # Assert result
    nodes_uuids = [
        root_node.uuid,
        child_node1.uuid,
        child_node3.uuid,
        grandchild_node2.uuid,
    ]
    assert all(node.uuid in nodes_uuids for node in result)
    assert child_node2.uuid not in nodes_uuids
    logger.info(result)


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

    await root_node.connect(child_node1)
    await root_node.connect(child_node2)
    await root_node.connect(child_node3)
    await child_node2.connect(grandchild_node1)
    await child_node3.connect(grandchild_node1)
    await grandchild_node1.connect(grandchild_node2)

    executor = AsyncTreeExecutor(
        uuid="Union Tree",
        roots=[root_node],
        use_dynamic_workers=False,
        num_workers=10,
    )
    result = await executor.run()

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
    assert len(result) == len(nodes_uuids)
    logger.info(result)


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

    await root_node.connect(child_node1)
    await root_node.connect(child_node2)
    await child_node1.connect(grandchild_node1)
    await child_node2.connect(grandchild_node2)

    executor = AsyncTreeExecutor(
        uuid="Error Tree", roots=[root_node], use_dynamic_workers=False, num_workers=10
    )
    result = await executor.run()

    # Assert result
    nodes_uuids = [root_node.uuid, child_node1.uuid, grandchild_node1.uuid]
    assert all(node.uuid in nodes_uuids for node in result)
    assert child_node2.uuid not in nodes_uuids
    assert grandchild_node2.uuid not in nodes_uuids
    logger.info(result)


@pytest.mark.asyncio
async def test_yielding():
    """
    Test the AsyncTreeExecutor's run_and_yield method to ensure it yields results as they are set.
    """
    root_node = create_node("root", mockup_coroutine)
    child_node1 = create_node("child1", mockup_coroutine)
    grandchild_node1 = create_node("grandchild1", mockup_coroutine)
    grandchild_node2 = create_node("grandchild2", mockup_coroutine)

    await root_node.connect(child_node1)
    await child_node1.connect(grandchild_node1)
    await child_node1.connect(grandchild_node2)

    # Manually connecting nodes
    executor = AsyncTreeExecutor(
        uuid="Yielding Tree", roots=[root_node], use_dynamic_workers=True
    )
    results = []

    async for node in executor.yielding():
        if not isinstance(node, Node):
            continue
        results.append((node.uuid, node))
        logger.info(f"Yielded: {node}")

    # Assert that all nodes have been processed and yielded
    nodes_uuids = [
        root_node.uuid,
        child_node1.uuid,
        grandchild_node1.uuid,
        grandchild_node2.uuid,
    ]
    yielded_uuids = [uuid for uuid, _ in results]
    assert all(node_uuid in yielded_uuids for node_uuid in nodes_uuids)
    logger.info("All nodes yielded successfully.")


@pytest.mark.asyncio
async def test_yield_with_timeout():
    """
    Test the AsyncTreeExecutor's yielding method with a UnionNode that times out,
    ensuring that nodes that exceed the timeout do not yield a result.
    """

    async def long_running_coroutine(node: Node):
        # Simulate a long-running task
        await asyncio.sleep(3)
        logger.info(f"{node.uuid} executed")
        return f"{node.uuid} result"

    root_node = create_node("root", mockup_coroutine)
    child1_node = create_node("child1", long_running_coroutine)
    child2_node = create_node("child2", mockup_coroutine)
    union_node = create_node(
        "union",
        long_running_coroutine,
        timeout=1,
    )

    await root_node.connect(child1_node)
    await root_node.connect(child2_node)
    await child1_node.connect(union_node)
    await child2_node.connect(union_node)

    executor = AsyncTreeExecutor(
        uuid="Yielding Tree with Timeout",
        roots=[root_node],
        use_dynamic_workers=False,
        num_workers=10,
    )

    results = []
    async for node in executor.yielding():
        if not isinstance(node, Node):
            continue
        results.append((node.uuid, node))
        logger.info(f"Yielded: {node}")

    expected_node_ids = [
        root_node.uuid,
        child1_node.uuid,
        child2_node.uuid,
    ]

    yielded_ids = [n_uuid for n_uuid, _ in results]
    assert all(node_uuid in yielded_ids for node_uuid in expected_node_ids)
    assert union_node.uuid not in yielded_ids
    logger.info(
        "Test yield with timeout: timed out union node did not yield result, others yielded successfully."
    )


@pytest.mark.asyncio
async def test_simple_tree_structure():
    """
    Test a simple tree structure with a root node, two children, and four grandchildren.
    """
    # Create nodes
    root_node = create_node("root", mockup_coroutine)
    child1_node = create_node("child1", mockup_coroutine)
    child2_node = create_node("child2", mockup_coroutine)
    grandchild1_node = create_node("grandchild1", mockup_coroutine)
    grandchild2_node = create_node("grandchild2", mockup_coroutine)
    grandchild3_node = create_node("grandchild3", mockup_coroutine)
    grandchild4_node = create_node("grandchild4", mockup_coroutine)

    await root_node.connect(child1_node)
    await root_node.connect(child2_node)
    await child1_node.connect(grandchild1_node)
    await child1_node.connect(grandchild2_node)
    await child2_node.connect(grandchild3_node)
    await child2_node.connect(grandchild4_node)

    # Create executor and build the tree
    executor = AsyncTreeExecutor(
        uuid="Simple Tree", roots=[root_node], use_dynamic_workers=True
    )
    result = await executor.run()

    # Assert all nodes were processed
    expected_nodes = [
        root_node.uuid,
        child1_node.uuid,
        child2_node.uuid,
        grandchild1_node.uuid,
        grandchild2_node.uuid,
        grandchild3_node.uuid,
        grandchild4_node.uuid,
    ]

    # Check that all expected nodes are in the result
    assert all(node.uuid in expected_nodes for node in result)
    assert len(result) == len(expected_nodes)

    logger.info(f"Simple tree test completed with {len(result)} nodes processed")


@pytest.mark.asyncio
async def test_multiple_roots_structure():
    """
    Test a tree structure with multiple root nodes, each with their own children.
    This tests the executor's ability to process trees without a single root node.
    """
    # Create nodes
    root1_node = create_node("root1", mockup_coroutine)
    root2_node = create_node("root2", mockup_coroutine)
    child1_node = create_node("child1", mockup_coroutine)
    child2_node = create_node("child2", mockup_coroutine)
    grandchild1_node = create_node("grandchild1", mockup_coroutine)
    grandchild2_node = create_node("grandchild2", mockup_coroutine)
    grandchild3_node = create_node("grandchild3", mockup_coroutine)
    grandchild4_node = create_node("grandchild4", mockup_coroutine)

    await root1_node.connect(child1_node)
    await root2_node.connect(child2_node)
    await child1_node.connect(grandchild1_node)
    await child1_node.connect(grandchild2_node)
    await child2_node.connect(grandchild3_node)
    await child2_node.connect(grandchild4_node)

    # Create executor and build the tree
    executor = AsyncTreeExecutor(
        uuid="Multiple Roots Tree",
        roots=[root1_node, root2_node],
        use_dynamic_workers=True,
    )
    result = await executor.run()

    # Assert all nodes were processed
    expected_nodes = [
        root1_node.uuid,
        root2_node.uuid,
        child1_node.uuid,
        child2_node.uuid,
        grandchild1_node.uuid,
        grandchild2_node.uuid,
        grandchild3_node.uuid,
        grandchild4_node.uuid,
    ]

    # Check that all expected nodes are in the result
    assert all(node.uuid in expected_nodes for node in result)
    assert len(result) == len(expected_nodes)

    logger.info(f"Multiple roots test completed with {len(result)} nodes processed")


@pytest.mark.asyncio
async def test_cycle():
    """
    Test a cycle in the tree structure with nodeA -> nodeB -> nodeA and then break the cycle.
    """
    # Create nodes
    node_a = create_node("nodeA", mockup_coroutine)
    node_b = create_node("nodeB", cycle_coroutine, kwargs={"child_node": node_a})

    # Connect nodes to form a cycle
    await node_a.connect(node_b)

    # Create executor and run the tree
    executor = AsyncTreeExecutor(uuid="Cycle Break Tree", roots=[node_a], num_workers=2)
    result = await executor.run()

    # Assert that the cycle was broken and nodes were processed
    nodes_uuids = [node_a.uuid, node_b.uuid]
    assert all(node.uuid in nodes_uuids for node in result)
    logger.info("Cycle break test completed with nodes processed successfully.")


@pytest.mark.asyncio
async def test_dynamic_cycle_connection():
    """
    Test dynamic cycle creation and breaking during runtime.
    Node A outputs random floats, Node B creates a cycle with A, lets A run again,
    then breaks the cycle.
    """
    total_a_runs = 0
    node_a_first_output = None
    node_a_second_output = None

    async def random_float_coroutine(node: Node, target_node: Node):
        """
        Coroutine that outputs a random float between 0 and 1.
        """
        nonlocal total_a_runs, node_a_first_output, node_a_second_output
        number = random.random()
        if total_a_runs > 0:
            await node.disconnect(target_node)
            node_a_second_output = number
        else:
            node_a_first_output = number

        print(f"{node.uuid} generated number: {number}")
        total_a_runs += 1
        return number

    async def cycle_creator_coroutine(node: Node, target_node: Node):
        """
        Coroutine that creates a cycle with the target node, waits for it to run,
        then breaks the cycle.
        """
        await node.connect(target_node)
        return f"{node.uuid} cycle completed"

    # Create nodes
    node_a = create_node("nodeA", random_float_coroutine)
    node_b = create_node("nodeB", cycle_creator_coroutine)
    node_a.kwargs = dict(node=node_a, target_node=node_b)
    node_b.kwargs = dict(node=node_b, target_node=node_a)

    # Initial connection A -> B
    await node_a.connect(node_b)

    # Create executor and run the tree
    executor = AsyncTreeExecutor(
        uuid="Dynamic Cycle Tree", roots=[node_a], num_workers=2
    )
    result = await executor.run()

    # Assert that both nodes were processed
    nodes_uuids = [node_a.uuid, node_b.uuid]
    assert all(node.uuid in nodes_uuids for node in result)

    # Verify that node A's first output is not equal to its second output
    assert node_a_first_output != node_a_second_output

    print("Dynamic cycle test completed successfully")


async def mockup_yielding_coroutine(node: Node):
    """
    Example async generator function that yields intermediate results.
    """
    for i in range(3):
        await asyncio.sleep(0.5)  # Simulate some work
        yield f"{node.uuid} progress {i}"

    # Final result
    await asyncio.sleep(0.5)
    yield f"{node.uuid} completed"


@pytest.mark.asyncio
async def test_mixed_tree_with_yielding():
    """
    Test a tree with mixed node types: some regular coroutines and some async generators.
    Uses manual .connect() to build the tree.
    """
    # Create nodes with mixed types
    root_node = create_node("root", mockup_coroutine)  # Regular coroutine
    child1_node = create_node("child1", mockup_yielding_coroutine)  # Async generator
    child2_node = create_node("child2", mockup_coroutine)  # Regular coroutine
    grandchild1_node = create_node(
        "grandchild1", mockup_yielding_coroutine
    )  # Async generator
    grandchild2_node = create_node("grandchild2", mockup_coroutine)  # Regular coroutine

    # Manually connect nodes to build the tree
    await root_node.connect(child1_node)
    await root_node.connect(child2_node)
    await child1_node.connect(grandchild1_node)
    await child2_node.connect(grandchild2_node)

    # Create executor and run the tree
    executor = AsyncTreeExecutor(uuid="Mixed Tree", roots=[root_node], num_workers=3)

    # Test the yielding method to get both node completions and intermediate results
    results = []
    node_completions = []
    intermediate_results: list[Chunk[str]] = []

    async for item in executor.yielding():
        if isinstance(item, Node):
            # This is a completed node
            node_completions.append(item.uuid)
            results.append(f"Completed: {item.uuid}")
            logger.info(f"Completed node: {item.uuid}")
        else:
            # This is an intermediate result from a yielding node
            intermediate_results.append(item)
            results.append(f"Result: {item.uuid} -> {item.output}")
            logger.info(f"Intermediate result: {item.uuid} -> {item.output}")

    # Assert that all nodes were completed
    expected_completed_nodes = [
        root_node.uuid,
        child1_node.uuid,
        child2_node.uuid,
        grandchild1_node.uuid,
        grandchild2_node.uuid,
    ]

    assert all(node_uuid in node_completions for node_uuid in expected_completed_nodes)
    assert len(node_completions) == len(expected_completed_nodes)

    # Assert that we got intermediate results from yielding nodes
    yielding_nodes = [child1_node.uuid, grandchild1_node.uuid]

    # Each yielding node should have yielded multiple results
    for yielding_node_uuid in yielding_nodes:
        node_results = [
            chunk.output
            for chunk in intermediate_results
            if chunk.uuid == yielding_node_uuid
        ]
        assert len(node_results) >= 3  # At least 3 yields per yielding node
        assert any("progress" in result for result in node_results)
        assert any("completed" in result for result in node_results)

    # Verify the structure of results
    assert len(results) > len(
        expected_completed_nodes
    )  # More results than nodes due to yielding

    logger.info("Mixed tree test completed successfully!")
    logger.info(f"Completed nodes: {node_completions}")
    logger.info(f"Intermediate results count: {len(intermediate_results)}")
    logger.info(f"Total results: {len(results)}")


@pytest.mark.asyncio
async def test_forwarding_success():
    """
    Test successful forwarding behavior where A -> B -> C, with A forwarding output to B properly,
    and B forwarding output to C without conflicts.
    """
    # Track forwarded values to verify the behavior
    forwarded_values = {}

    async def node_a_coroutine():
        """Node A produces a value and forwards it to B."""
        result = "data_from_A"
        forwarded_values["A"] = result
        return result

    async def node_b_coroutine(data_from_A: str):
        """Node B receives data from A, processes it, and forwards to C."""
        # Verify B received the forwarded data from A
        assert data_from_A == "data_from_A"
        result = f"processed_{data_from_A}"
        forwarded_values["B"] = result
        return result

    async def node_c_coroutine(data_from_B: str, existing_value: str):
        """Node C receives data from B without conflicts."""
        # Verify C received the forwarded data from B
        assert data_from_B == "processed_data_from_A"
        # The existing_value should remain unchanged (it's a different parameter)
        assert existing_value == "original_value"
        result = f"final_{data_from_B}"
        forwarded_values["C"] = result
        return result

    # Create nodes without forwarding configuration in constructor
    node_a = Node(uuid="nodeA", coroutine=node_a_coroutine)
    node_b = Node(uuid="nodeB", coroutine=node_b_coroutine)
    node_c = Node(uuid="nodeC", coroutine=node_c_coroutine)

    # Set up C with only non-conflicting values
    node_c.kwargs["existing_value"] = "original_value"

    # Connect the nodes with forwarding: A -> B -> C
    await node_a.connect(node_b, forward_as="data_from_A")
    await node_b.connect(node_c, forward_as="data_from_B")

    # Create executor and run the tree
    executor = AsyncTreeExecutor(
        uuid="Forwarding Success Test", roots=[node_a], num_workers=3
    )
    result = await executor.run()

    # Assert all nodes were processed
    expected_nodes = [node_a.uuid, node_b.uuid, node_c.uuid]
    assert all(node.uuid in expected_nodes for node in result)
    assert len(result) == len(expected_nodes)

    # Verify the forwarding chain worked correctly
    assert forwarded_values["A"] == "data_from_A"
    assert forwarded_values["B"] == "processed_data_from_A"
    assert forwarded_values["C"] == "final_processed_data_from_A"

    # Verify the final state of node C's kwargs
    assert node_c.kwargs["data_from_B"] == "processed_data_from_A"
    assert node_c.kwargs["existing_value"] == "original_value"


@pytest.mark.asyncio
async def test_forwarding_conflict_error():
    """
    Test forwarding behavior where a conflict occurs when trying to forward to a child
    that already has an argument with the same name.
    """
    # Track forwarded values to verify the behavior
    forwarded_values = {}

    async def node_a_coroutine():
        """Node A produces a value and forwards it to B."""
        result = "data_from_A"
        forwarded_values["A"] = result
        return result

    async def node_b_coroutine(data_from_A: str):
        """Node B receives data from A, processes it, and forwards to C."""
        # Verify B received the forwarded data from A
        assert data_from_A == "data_from_A"
        result = f"processed_{data_from_A}"
        forwarded_values["B"] = result
        return result

    async def node_c_coroutine(data_from_B: str, existing_value: str):
        """Node C receives data from B without conflicts."""
        # Verify C received the forwarded data from B
        assert data_from_B == "processed_data_from_A"
        # The existing_value should remain unchanged (it's a different parameter)
        assert existing_value == "original_value"
        result = f"final_{data_from_B}"
        forwarded_values["C"] = result
        return result

    # Create nodes without forwarding configuration in constructor
    node_a = Node(uuid="nodeA", coroutine=node_a_coroutine)
    node_b = Node(uuid="nodeB", coroutine=node_b_coroutine)
    node_c = Node(uuid="nodeC", coroutine=node_c_coroutine)

    # Set up C with a value that will cause a conflict
    node_c.kwargs["data_from_B"] = "will_raise_error"

    # Connect the nodes with forwarding: A -> B -> C
    await node_a.connect(node_b, forward_as="data_from_A")
    await node_b.connect(node_c, forward_as="data_from_B")

    # Create executor and run the tree
    executor = AsyncTreeExecutor(
        uuid="Forwarding Conflict Test", roots=[node_a], num_workers=3
    )
    result = await executor.run()

    # Assert all nodes were processed
    expected_nodes = [node_a.uuid, node_b.uuid, node_c.uuid]
    assert all(node.uuid in expected_nodes for node in result)
    assert len(result) == 1

    # Verify the forwarding chain worked correctly
    assert forwarded_values["A"] == "data_from_A"
    assert forwarded_values["B"] == "processed_data_from_A"
    assert "C" not in forwarded_values.keys()

    # Verify the final state of node C's kwargs
    assert node_c.kwargs["data_from_B"] == "will_raise_error"
