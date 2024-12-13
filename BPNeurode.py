"""
This neurode computes deltas, changes weights and passes back error info.

Every neurode has its delta, which stands for an error.
"""

from Neurode import Neurode


class BPNeurode(Neurode):
    """The BPNNeurode class deals with the backwards phase of training.

    For every neurode calculates delta and performs the weight update.
    """

    def __init__(self):
        """Call parent init first."""
        super().__init__()
        self._delta = 0
        self._weights = {}

    @staticmethod
    def _sigmoid_derivative(value):
        """f(x) * (1-f(x)) forumula."""
        return value * (1 - value)

    def _calculate_delta(self, expected_value=None):
        """Output layer case."""
        if expected_value is not None:
            error = expected_value - self._value
            self._delta = error * self._sigmoid_derivative(self._value)
            return

        """Hidden layer, loop through downstream nodes"""
        total = 0
        for next_node in self._neighbors[self.Side.DOWNSTREAM]:
            """Figure out which input I am to the next node"""
            total += next_node._weights[self] * next_node.delta

        self._delta = total * self._sigmoid_derivative(self._value)

    def data_ready_downstream(self, node):
        """Responds to data from downstream node during backdrop."""
        if self._check_in(node, self.Side.DOWNSTREAM):
            """Calculate delta first or weights get messed up"""
            self._calculate_delta()
            self._fire_upstream()
            self._update_weights()

    def set_expected(self, expected_value):
        """Output layer gets its delta from expected value."""
        self._calculate_delta(expected_value)
        """ Tell upstream nodes we got data """
        for node in self._neighbors[self.Side.UPSTREAM]:
            node.data_ready_downstream(self)

    def adjust_weights(self, node, adjustment):
        """Find the right weight to change."""
        self._weights[node] = self._weights.get(node, 0) + adjustment

    def _update_weights(self):
        """Weight change formula from class notes."""
        for next_node in self._neighbors[self.Side.DOWNSTREAM]:
            weight_change = self.learning_rate * self._value * next_node.delta
            next_node.adjust_weights(self, weight_change)

    def _fire_upstream(self):
        for node in self._neighbors[self.Side.UPSTREAM]:
            node.data_ready_downstream(self)

    @property
    def delta(self):
        """Get the delta value for this neurode."""
        return self._delta
