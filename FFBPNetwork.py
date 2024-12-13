"""
Use LayerList to create a network.

Provide training and testing methods that take advantage of the NNData.
"""

from LayerList import LayerList
from FFBPNeurode import FFBPNeurode
import NNData
from RMSE import RMSE


class FFBPNetwork:
    """Use LayerList to create a network."""

    class EmptySetException(Exception):
        """Notify client that there is no data loaded in the network."""

        pass

    def __init__(self, num_inputs: int, num_outputs: int,
                 error_model: type(RMSE)):
        """Set up a Neural Network with initial input and output neurodes.

        :param int num_inputs: Number of input layer neurodes.
        :param int num_outputs: Number of output layer neurodes.
        :param type(RMSE) error_model: Error model to use when reporting
            RMSE
        """
        self.layers = LayerList(num_inputs, num_outputs, FFBPNeurode)
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._errors = error_model()

    def add_hidden_layer(self, num_nodes, position=0):
        """Add a hidden layer to the network.

        :param int num_nodes: Number of neurodes to populate in the new
            hidden layer.
        :param int position: Location to insert layer, with 0 indicating
            that the new layer should be the first hidden layer, etc.
            (default is 0).
        """
        self.layers.reset_to_head()
        for _ in range(position):
            self.layers.move_forward()
        self.layers.add_layer(num_nodes)

    def train(self, data_set: NNData, epochs=1000,
              verbosity=2, order=NNData.Order.SHUFFLE):
        """Train the network for a number of epochs.

        :param NNData data_set: An NNData object with a dataset loaded.
        :param int epochs: Number of epochs to train (default is 1000).
        :param int verbosity: How much information to display during
            training. 0=only final RMSE, >0 RMSE every 100 epochs,
            >1 all predicted and expected values every 1000 epochs
            (default is 2).
        :param NNData.Order order: Whether to shuffle examples prior
            to training each epoch (default is NNData.Order.SHUFFLE).
        """
        if data_set.number_of_samples(NNData.Set.TRAIN) == 0:
            raise FFBPNetwork.EmptySetException
        for epoch in range(0, epochs):
            self._errors.reset()
            data_set.prime_data(order=order)
            while not data_set.pool_is_empty(NNData.Set.TRAIN):
                x, y = data_set.get_one_item(NNData.Set.TRAIN)
                zipped = zip(x, self.layers.input_nodes)
                for val, node in zipped:
                    node.set_input(val)
                predicted = []
                zipped = zip(y, self.layers.output_nodes)
                for val, node in zipped:
                    node.set_expected(val)
                    predicted.append(node.value)
                self._errors += (y, predicted)
                if epoch % 1000 == 0 and verbosity > 1:
                    print("Sample", x, "expected", y, "predicted", predicted)
            if epoch % 100 == 0 and verbosity > 0:
                print(f"Epoch {epoch} RMSE = {self._errors.error}")
        print(f"Final Training RMSE = {self._errors.error}")

    def test(self, data_set: NNData, order=NNData.Order.STATIC):
        """Test the network.

        :param NNData data_set: An NNData object with a dataset loaded.
        :param NNData.Order order: Whether to shuffle examples prior
            to testing (default is NNData.Order.STATIC).
        """
        if data_set.number_of_samples(NNData.Set.TEST) == 0:
            raise FFBPNetwork.EmptySetException
        self._errors.reset()
        data_set.prime_data(order=order)
        while not data_set.pool_is_empty(NNData.Set.TEST):
            x, y = data_set.get_one_item(NNData.Set.TEST)
            for j, node in enumerate(self.layers.input_nodes):
                node.set_input(x[j])
            predicted = []
            for j, node in enumerate(self.layers.output_nodes):
                predicted.append(node.value)
            self._errors += (y, predicted)
            print(f"{x}, {y}, {predicted}")
        print("RMSE = ", self._errors.error)
