"""Neural Network Node Implementation."""
from __future__ import annotations
from enum import Enum
import random
from typing import Dict, List


class MultiLinkNode:
    """Base class for neural network nodes. Tracks node reports with binary."""

    class Side(Enum):
        """Defines which side nodes connect to."""

        UPSTREAM = 1    # nodes feeding in
        DOWNSTREAM = 2  # nodes feeding out

    def __init__(self):
        """Initialize tracking dictionaries and neighbor lists."""
        # track which nodes reported in (using binary)
        self._reporting_nodes = {
            self.Side.UPSTREAM: 0,
            self.Side.DOWNSTREAM: 0
        }

        # what binary number we expect when all nodes report in
        self._reference_value = {
            self.Side.UPSTREAM: 0,
            self.Side.DOWNSTREAM: 0
        }

        # lists of nodes we're connected to
        self._neighbors: Dict[MultiLinkNode.Side, List[MultiLinkNode]] = {
            self.Side.UPSTREAM: [],
            self.Side.DOWNSTREAM: []
        }

    def __str__(self) -> str:
        """Show the node's connections."""
        ups = self._neighbors[self.Side.UPSTREAM]
        downs = self._neighbors[self.Side.DOWNSTREAM]
        up_nodes = [str(id(n)) for n in ups]
        down_nodes = [str(id(n)) for n in downs]
        out = f"Node {id(self)}:\n"
        out += f"Upstream: {up_nodes}\n"
        out += f"Downstream: {down_nodes}"
        return out

    def _process_new_neighbor(
            self, node: MultiLinkNode, side: MultiLinkNode.Side
    ) -> None:
        """Child classes need to handle new neighbors their own way."""
        raise NotImplementedError("Implement in child class")

    def reset_neighbors(self, nodes: list, side: MultiLinkNode.Side) -> None:
        """Reset and setup new neighbor connections."""
        # copy the list so changes won't mess us up
        self._neighbors[side] = nodes.copy()

        # each node gets a binary position (like 1, 2, 4, 8...)
        # add them up to get expected value when all report in
        ref_val = 0
        for i in range(len(nodes)):
            ref_val += 2**i
        self._reference_value[side] = ref_val

        # setup each connection
        for node in nodes:
            self._process_new_neighbor(node, side)


class Neurode(MultiLinkNode):
    """Node for neural network. Has weights for upstream connections."""

    _learning_rate = 0.05  # same rate for all nodes

    def __init__(self):
        """Setups new node with zero value and empty weights."""
        super().__init__()
        self._value = 0  # node's current value
        self._weights = {}  # store weights for upstream nodes

    @property
    def learning_rate(self) -> float:
        """Get current learning rate."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, rate: float) -> None:
        """Update learning rate for all nodes."""
        Neurode._learning_rate = rate

    @property
    def value(self) -> float:
        """Get node's current value."""
        return self._value

    def _process_new_neighbor(
            self, node: Neurode, side: MultiLinkNode.Side
    ) -> None:
        """Give random weight to new upstream neighbors."""
        if side == self.Side.UPSTREAM:
            self._weights[node] = random.random()  # random float 0 to 1

    def get_weight(self, node: Neurode) -> float:
        """Get the weight for a given upstream node."""
        return self._weights[node]

    def _check_in(self, node: Neurode, side: MultiLinkNode.Side) -> bool:
        """Track which nodes have reported in using binary math."""
        # find node's position in list
        node_pos = self._neighbors[side].index(node)

        # use bit shift to mark this node as reported
        self._reporting_nodes[side] |= (1 << node_pos)

        # all reported in if we match reference
        if self._reporting_nodes[side] == self._reference_value[side]:
            self._reporting_nodes[side] = 0  # reset for next round
            return True

        return False
