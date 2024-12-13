"""
A list implementation specialized for managing layers of a neural network.

Which was developed on top of DoublyLinkedList
to manage organization and connections.
between layers of neurodes.
"""

from DoublyLinkedList import DoublyLinkedList
from Neurode import MultiLinkNode


class LayerList(DoublyLinkedList):
    """Class for managing layers of neurodes in a neural network."""

    def __init__(self, inputs: int, outputs: int, neurode_type):
        """Initialize the network with input and output layers."""
        super().__init__()
        self._neurode_type = neurode_type

        input_layer = []
        for _ in range(inputs):
            input_layer.append(neurode_type())

        output_layer = []
        for _ in range(outputs):
            output_layer.append(neurode_type())

        self._connect_layers(input_layer, output_layer)
        self.add_to_head(input_layer)
        self.add_after_current(output_layer)

    def _connect_layers(self, layer1, layer2):
        """Connect each node in layer1 to each node in layer2."""
        side = MultiLinkNode.Side
        for node1 in layer1:
            node1.reset_neighbors(layer2, side.DOWNSTREAM)
        for node2 in layer2:
            node2.reset_neighbors(layer1, side.UPSTREAM)

    def _disconnect_layers(self, layer1, layer2):
        """Remove all connections between layer1 and layer2."""
        for node in layer1:
            node.reset_neighbors([], True)
        for node in layer2:
            node.reset_neighbors([], False)

    def add_layer(self, num_nodes: int):
        """Add a hidden layer after the current position."""
        if self._curr.next is None:
            raise IndexError("Cannot add layer after output layer")

        new_nodes = []
        for _ in range(num_nodes):
            new_nodes.append(self._neurode_type())

        current_layer = self._curr.data
        next_layer = self._curr.next.data

        self._disconnect_layers(current_layer, next_layer)
        self._connect_layers(current_layer, new_nodes)
        self._connect_layers(new_nodes, next_layer)

        self.add_after_current(new_nodes)

    def remove_layer(self):
        """Remove the layer after the current position."""
        if self._curr.next is None or self._curr.next.next is None:
            raise IndexError("Cannot remove output layer")

        layer_before = self._curr.data
        goner = self._curr.next.data
        layer_after = self._curr.next.next.data

        self._disconnect_layers(layer_before, goner)
        self._disconnect_layers(goner, layer_after)
        self._connect_layers(layer_before, layer_after)

        self.remove_after_current()

    @property
    def input_nodes(self):
        """Return the input layer neurodes."""
        return self.head.data

    @property
    def output_nodes(self):
        """Return the output layer neurodes."""
        return self._tail.data
