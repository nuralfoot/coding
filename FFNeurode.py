"""
Development of a feedforward neural network node.

Based on the Neurode class. Every node is connected to other
nodes in the upstream and downstream manner.
"""

from __future__ import annotations  # needed for type hints
from Neurode import Neurode
import math  # for exp function


class FFNeurode(Neurode):
    """Feed forward neurode class that inherits from base Neurode."""

    @staticmethod
    def _sigmoid(value):
        # does the sigmoid calculation
        result = 1 / (1 + math.exp(-value))
        return result

    def _calculate_value(self):
        # calculate the weighted sum and put through sigmoid
        total = 0
        # loop through upstream nodes
        for curr_node in self._neighbors[self.Side.UPSTREAM]:
            # multiply value by weight and add to total
            curr_weight = self._weights[curr_node]
            curr_val = curr_node.value
            total += curr_val * curr_weight

        # set node value using sigmoid
        self._value = self._sigmoid(total)

    def _fire_downstream(self):
        # tell downstream nodes we have data
        downstream_nodes = self._neighbors[self.Side.DOWNSTREAM]
        for n in downstream_nodes:
            # call data ready on each node
            n.data_ready_upstream(self)

    def data_ready_upstream(self, node):
        """Upstream node has data ready."""
        # check if all nodes reported in
        all_reported = self._check_in(node, self.Side.UPSTREAM)

        if all_reported is True:
            # if all nodes reported, calculate new value
            self._calculate_value()
            # and tell downstream nodes
            self._fire_downstream()

    def set_input(self, input_value):
        """Set value for input layer nodes."""
        self._value = input_value  # set the value
        # tell downstream nodes data is ready
        self._fire_downstream()
