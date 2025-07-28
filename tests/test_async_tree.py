import asyncio
from grafo.components import Chunk
import logging
import random
from typing import Any, Callable, Optional

import pytest

from grafo._internal import logger
from grafo import TreeExecutor, Node

logger.setLevel(logging.DEBUG)


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

    executor = TreeExecutor(uuid="Manual Tree", roots=[root_node])
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

    await root_node.connect(child_node1)
    await root_node.connect(child_node2)
    await root_node.connect(child_node3)

    executor = TreeExecutor(uuid="Picker Tree", roots=[root_node])
    result = await executor.run()

    # Assert result
    nodes_uuids = [
        root_node.uuid,
        child_node1.uuid,
        child_node3.uuid,
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

    executor = TreeExecutor(uuid="Union Tree", roots=[root_node])
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

    executor = TreeExecutor(uuid="Error Tree", roots=[root_node])
    result = await executor.run()

    # Assert result
    nodes_uuids = [root_node.uuid, child_node1.uuid, grandchild_node1.uuid]
    assert all(node.uuid in nodes_uuids for node in result)
    assert child_node2.uuid not in nodes_uuids
    assert grandchild_node2.uuid not in nodes_uuids


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
    executor = TreeExecutor(uuid="Yielding Tree", roots=[root_node])
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

    executor = TreeExecutor(uuid="Yielding Tree with Timeout", roots=[root_node])

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
    executor = TreeExecutor(uuid="Simple Tree", roots=[root_node])
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
    executor = TreeExecutor(uuid="Multiple Roots Tree", roots=[root1_node, root2_node])
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
    executor = TreeExecutor(uuid="Cycle Break Tree", roots=[node_a])
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
    executor = TreeExecutor(uuid="Dynamic Cycle Tree", roots=[node_a])
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
    executor = TreeExecutor(uuid="Mixed Tree", roots=[root_node])

    # Test the yielding method to get both node completions and intermediate results
    results = []
    node_completions = []
    intermediate_results: list[Chunk[str]] = []

    async for item in executor.yielding():
        node_completions.append(item.output)
        if isinstance(item, Node):
            # This is a completed node
            results.append(f"Completed: {item.uuid}")
            logger.info(f"Completed node: {item.uuid}")
        else:
            # This is an intermediate result from a yielding node
            intermediate_results.append(item)
            results.append(f"Result: {item.uuid} -> {item.output}")
            logger.info(f"Intermediate result: {item.uuid} -> {item.output}")

    # Assert that all nodes were completed
    expected_results = [
        "root result",
        "child1 progress 0",
        "child2 result",
        "child1 progress 1",
        "child1 progress 2",
        "child1 completed",
        "grandchild2 result",
        "grandchild1 progress 0",
        "grandchild1 progress 1",
        "grandchild1 progress 2",
        "grandchild1 completed",
    ]

    assert all(node_uuid in node_completions for node_uuid in expected_results)

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
    executor = TreeExecutor(uuid="Forwarding Success Test", roots=[node_a])
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
    executor = TreeExecutor(uuid="Forwarding Conflict Test", roots=[node_a])
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


@pytest.mark.asyncio
async def test_on_before_forward_filtering():
    """
    Test using on_before_forward to filter and forward different parts of node A's output
    to different children. Node A outputs two numbers, and each child receives only one
    of them based on the filtering logic.
    """
    # Track forwarded values to verify the behavior
    forwarded_values = {}

    async def node_a_coroutine():
        """Node A produces a tuple of two numbers."""
        result = (42, 100)
        forwarded_values["A"] = result
        return result

    async def node_b_coroutine(first_number: int):
        """Node B receives only the first number from A."""
        # Verify B received only the first number
        assert first_number == 42
        result = f"B processed: {first_number}"
        forwarded_values["B"] = result
        return result

    async def node_c_coroutine(second_number: int):
        """Node C receives only the second number from A."""
        # Verify C received only the second number
        assert second_number == 100
        result = f"C processed: {second_number}"
        forwarded_values["C"] = result
        return result

    # Filter functions for on_before_forward
    async def filter_first_number(forward_data: tuple[int, int]) -> int:
        """Extract only the first number from the tuple."""
        first_num, _ = forward_data
        return first_num

    async def filter_second_number(forward_data: tuple[int, int]) -> int:
        """Extract only the second number from the tuple."""
        _, second_num = forward_data
        return second_num

    # Create nodes
    node_a = Node(uuid="nodeA", coroutine=node_a_coroutine)
    node_b = Node(uuid="nodeB", coroutine=node_b_coroutine)
    node_c = Node(uuid="nodeC", coroutine=node_c_coroutine)

    # Connect A to B with filtering for first number
    await node_a.connect(
        node_b,
        forward_as="first_number",
        on_before_forward=(filter_first_number, None),
    )

    # Connect A to C with filtering for second number
    await node_a.connect(
        node_c, forward_as="second_number", on_before_forward=(filter_second_number, {})
    )

    # Create executor and run the tree
    executor = TreeExecutor(uuid="On Before Forward Filtering Test", roots=[node_a])
    result = await executor.run()

    # Assert all nodes were processed
    expected_nodes = [node_a.uuid, node_b.uuid, node_c.uuid]
    assert all(node.uuid in expected_nodes for node in result)
    assert len(result) == len(expected_nodes)

    # Verify the forwarding chain worked correctly
    assert forwarded_values["A"] == (42, 100)
    assert forwarded_values["B"] == "B processed: 42"
    assert forwarded_values["C"] == "C processed: 100"

    # Verify the final state of each node's kwargs
    assert node_b.kwargs["first_number"] == 42
    assert node_c.kwargs["second_number"] == 100

    logger.info("On before forward filtering test completed successfully!")


@pytest.mark.asyncio
async def test_on_before_forward_with_kwargs():
    """
    Test using on_before_forward with additional kwargs to demonstrate
    more complex filtering scenarios.
    """
    # Track forwarded values to verify the behavior
    forwarded_values = {}

    async def node_a_coroutine():
        """Node A produces a list of numbers."""
        result = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        forwarded_values["A"] = result
        return result

    async def node_b_coroutine(even_numbers: list[int]):
        """Node B receives only even numbers."""
        # Verify B received only even numbers
        assert all(num % 2 == 0 for num in even_numbers)
        result = f"B processed {len(even_numbers)} even numbers: {even_numbers}"
        forwarded_values["B"] = result
        return result

    async def node_c_coroutine(odd_numbers: list[int]):
        """Node C receives only odd numbers."""
        # Verify C received only odd numbers
        assert all(num % 2 == 1 for num in odd_numbers)
        result = f"C processed {len(odd_numbers)} odd numbers: {odd_numbers}"
        forwarded_values["C"] = result
        return result

    # Filter functions with kwargs
    async def filter_even_numbers(forward_data: list[int], **kwargs) -> list[int]:
        """Filter even numbers from the list."""
        max_count = kwargs.get("max_count", len(forward_data))
        even_nums = [num for num in forward_data if num % 2 == 0]
        return even_nums[:max_count]

    async def filter_odd_numbers(forward_data: list[int], **kwargs) -> list[int]:
        """Filter odd numbers from the list."""
        max_count = kwargs.get("max_count", len(forward_data))
        odd_nums = [num for num in forward_data if num % 2 == 1]
        return odd_nums[:max_count]

    # Create nodes
    node_a = Node(uuid="nodeA", coroutine=node_a_coroutine)
    node_b = Node(uuid="nodeB", coroutine=node_b_coroutine)
    node_c = Node(uuid="nodeC", coroutine=node_c_coroutine)

    # Connect A to B with filtering for even numbers (max 3)
    await node_a.connect(
        node_b,
        forward_as="even_numbers",
        on_before_forward=(filter_even_numbers, {"max_count": 3}),
    )

    # Connect A to C with filtering for odd numbers (max 2)
    await node_a.connect(
        node_c,
        forward_as="odd_numbers",
        on_before_forward=(filter_odd_numbers, {"max_count": 2}),
    )

    # Create executor and run the tree
    executor = TreeExecutor(uuid="On Before Forward With Kwargs Test", roots=[node_a])
    result = await executor.run()

    # Assert all nodes were processed
    expected_nodes = [node_a.uuid, node_b.uuid, node_c.uuid]
    assert all(node.uuid in expected_nodes for node in result)
    assert len(result) == len(expected_nodes)

    # Verify the forwarding chain worked correctly
    assert forwarded_values["A"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert forwarded_values["B"] == "B processed 3 even numbers: [2, 4, 6]"
    assert forwarded_values["C"] == "C processed 2 odd numbers: [1, 3]"

    # Verify the final state of each node's kwargs
    assert node_b.kwargs["even_numbers"] == [2, 4, 6]
    assert node_c.kwargs["odd_numbers"] == [1, 3]

    logger.info("On before forward with kwargs test completed successfully!")


@pytest.mark.asyncio
async def test_repr_two_roots_conjoined():
    """
    Test the __repr__ method of AsyncTreeExecutor with a tree that has 2 roots
    that eventually become conjoined through a shared child node.
    """
    # Create nodes for the first root branch
    root1_node = create_node("root1", mockup_coroutine)
    child1_node = create_node("child1", mockup_coroutine)
    grandchild1_node = create_node("grandchild1", mockup_coroutine)

    # Create nodes for the second root branch
    root2_node = create_node("root2", mockup_coroutine)
    child2_node = create_node("child2", mockup_coroutine)

    # Create the shared node where the two branches conjoin
    shared_node = create_node("shared", mockup_coroutine)
    final_node = create_node("final", mockup_coroutine)

    # Connect first branch: root1 -> child1 -> grandchild1 -> shared
    await root1_node.connect(child1_node)
    await child1_node.connect(grandchild1_node)
    await grandchild1_node.connect(shared_node)

    # Connect second branch: root2 -> child2 -> shared
    await root2_node.connect(child2_node)
    await child2_node.connect(shared_node)

    # Connect shared node to final node
    await shared_node.connect(final_node)

    # Create executor with both roots
    executor = TreeExecutor(
        uuid="Conjoined Tree Test",
        description="A tree with two roots that conjoin at a shared node",
        roots=[root1_node, root2_node],
    )

    # Get the string representation
    tree_repr = repr(executor)

    # Print the representation for visual verification
    print("\n" + "=" * 50)
    print("TREE REPRESENTATION:")
    print("=" * 50)
    print(tree_repr)
    print("=" * 50)

    # Assert the representation contains expected elements
    assert "UUID: Conjoined Tree Test" in tree_repr
    assert (
        "Description: A tree with two roots that conjoin at a shared node" in tree_repr
    )
    assert "Structure:" in tree_repr

    # Assert both roots are present
    assert "Root root1:" in tree_repr
    assert "Root root2:" in tree_repr

    # Assert the connections from root1 branch
    assert "root1 -> child1" in tree_repr
    assert "child1 -> grandchild1" in tree_repr
    assert "grandchild1 -> shared" in tree_repr

    # Assert the connections from root2 branch
    assert "root2 -> child2" in tree_repr
    assert "child2 -> shared" in tree_repr

    # Assert the shared node connections
    assert "shared -> final" in tree_repr

    # Verify the structure is correct by running the tree
    result = await executor.run()

    # Assert all nodes were processed
    expected_nodes = [
        root1_node.uuid,
        root2_node.uuid,
        child1_node.uuid,
        child2_node.uuid,
        grandchild1_node.uuid,
        shared_node.uuid,
        final_node.uuid,
    ]

    assert all(node.uuid in expected_nodes for node in result)
    assert len(result) == len(expected_nodes)

    logger.info("Conjoined tree representation test completed successfully!")


@pytest.mark.asyncio
async def test_get_output_nodes():
    """
    Test the get_output_nodes method of AsyncTreeExecutor to ensure it correctly
    identifies all leaf nodes (nodes with no children) in the tree.
    """
    # Create a complex tree structure with multiple leaf nodes
    root_node = create_node("root", mockup_coroutine)

    # First branch: root -> child1 -> leaf1
    child1_node = create_node("child1", mockup_coroutine)
    leaf1_node = create_node("leaf1", mockup_coroutine)

    # Second branch: root -> child2 -> leaf2, leaf3
    child2_node = create_node("child2", mockup_coroutine)
    leaf2_node = create_node("leaf2", mockup_coroutine)
    leaf3_node = create_node("leaf3", mockup_coroutine)

    # Third branch: root -> child3 -> grandchild -> leaf4
    child3_node = create_node("child3", mockup_coroutine)
    grandchild_node = create_node("grandchild", mockup_coroutine)
    leaf4_node = create_node("leaf4", mockup_coroutine)

    # Fourth branch: root -> leaf5 (direct leaf)
    leaf5_node = create_node("leaf5", mockup_coroutine)

    # Connect the tree
    await root_node.connect(child1_node)
    await root_node.connect(child2_node)
    await root_node.connect(child3_node)
    await root_node.connect(leaf5_node)

    await child1_node.connect(leaf1_node)
    await child2_node.connect(leaf2_node)
    await child2_node.connect(leaf3_node)
    await child3_node.connect(grandchild_node)
    await grandchild_node.connect(leaf4_node)

    # Create executor
    executor = TreeExecutor(
        uuid="Output Nodes Test",
        description="Testing get_output_nodes method",
        roots=[root_node],
    )

    # Get the output nodes (leaf nodes)
    output_nodes = executor.get_leaves()

    # Print the output nodes for verification
    print("\n" + "=" * 50)
    print("OUTPUT NODES (LEAF NODES):")
    print("=" * 50)
    for node in output_nodes:
        print(f"Leaf node: {node.uuid}")
    print("=" * 50)

    # Expected leaf nodes
    expected_leaf_nodes = [
        leaf1_node.uuid,
        leaf2_node.uuid,
        leaf3_node.uuid,
        leaf4_node.uuid,
        leaf5_node.uuid,
    ]

    # Assert we got the correct number of leaf nodes
    assert len(output_nodes) == len(expected_leaf_nodes)

    # Assert all expected leaf nodes are present
    output_node_uuids = [node.uuid for node in output_nodes]
    assert all(leaf_uuid in output_node_uuids for leaf_uuid in expected_leaf_nodes)

    # Assert that non-leaf nodes are NOT in the output
    non_leaf_nodes = [
        root_node.uuid,
        child1_node.uuid,
        child2_node.uuid,
        child3_node.uuid,
        grandchild_node.uuid,
    ]
    assert all(
        non_leaf_uuid not in output_node_uuids for non_leaf_uuid in non_leaf_nodes
    )

    # Test with multiple roots
    root2_node = create_node("root2", mockup_coroutine)
    leaf6_node = create_node("leaf6", mockup_coroutine)
    await root2_node.connect(leaf6_node)

    executor_multi_root = TreeExecutor(
        uuid="Multi-Root Output Nodes Test",
        description="Testing get_output_nodes with multiple roots",
        roots=[root_node, root2_node],
    )

    # Get output nodes for multi-root tree
    multi_output_nodes = executor_multi_root.get_leaves()

    # Expected leaf nodes for multi-root tree
    expected_multi_leaf_nodes = expected_leaf_nodes + [leaf6_node.uuid]

    # Assert we got the correct number of leaf nodes
    assert len(multi_output_nodes) == len(expected_multi_leaf_nodes)

    # Assert all expected leaf nodes are present
    multi_output_node_uuids = [node.uuid for node in multi_output_nodes]
    assert all(
        leaf_uuid in multi_output_node_uuids for leaf_uuid in expected_multi_leaf_nodes
    )

    # Test with a tree that has a single node (root is also a leaf)
    single_node = create_node("single", mockup_coroutine)
    executor_single = TreeExecutor(
        uuid="Single Node Test",
        description="Testing get_output_nodes with a single node",
        roots=[single_node],
    )

    single_output_nodes = executor_single.get_leaves()
    assert len(single_output_nodes) == 1
    assert single_output_nodes[0].uuid == single_node.uuid

    logger.info("get_output_nodes test completed successfully!")
