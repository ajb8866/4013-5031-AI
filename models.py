### code base: ai.berkeley.edu

import nn


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 25             # 10
        self.hidden_layer_size = 350     # 350
        self.num_labels = 10

        # hidden layer 1
        self.w_1 = nn.Parameter(784, self.hidden_layer_size)
        self.b_1 = nn.Parameter(1, self.hidden_layer_size)

        # hidden layer 2
        self.w_2 = nn.Parameter(self.hidden_layer_size, self.hidden_layer_size)
        self.b_2 = nn.Parameter(1, self.hidden_layer_size)

        # hidden layer 3
        self.w_3 = nn.Parameter(self.hidden_layer_size, self.hidden_layer_size)
        self.b_3 = nn.Parameter(1, self.hidden_layer_size)

        # output vector
        self.output_wt = nn.Parameter(self.hidden_layer_size, self.num_labels)
        self.output_bias = nn.Parameter(1, self.num_labels)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
         # hidden layer 1
        trans_1 = nn.Linear(x, self.w_1)
        trans_bias_1 = nn.AddBias(trans_1, self.b_1)
        layer_1 = nn.ReLU(trans_bias_1)

        # hidden layer 2
        trans_2 = nn.Linear(layer_1, self.w_2)
        trans_bias_2 = nn.AddBias(trans_2, self.b_2)
        layer_2 = nn.ReLU(trans_bias_2)

        # hidden layer 3
        trans_3 = nn.Linear(layer_2, self.w_3)
        trans_bias_3 = nn.AddBias(trans_3, self.b_3)
        layer_3 = nn.ReLU(trans_bias_3)

        # output vector (no relu)
        last_trans = nn.Linear(layer_3, self.output_wt)
        return nn.AddBias(last_trans, self.output_bias)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_hats = self.run(x)
        return nn.SoftmaxLoss(y_hats, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        adjusted_rate = -0.12
        while True:

            for row_vect, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(row_vect, y)
                params = ([self.w_1, self.w_2, self.w_3, self.output_wt,
                           self.b_1, self.b_2, self.b_3, self.output_bias])
                gradients = nn.gradients(loss, params)
                learning_rate = min(-0.005, adjusted_rate)

                # updates
                self.w_1.update(gradients[0], learning_rate)
                self.w_2.update(gradients[1], learning_rate)
                self.w_3.update(gradients[2], learning_rate)
                self.output_wt.update(gradients[3], learning_rate)
                self.b_1.update(gradients[4], learning_rate)
                self.b_2.update(gradients[5], learning_rate)
                self.b_3.update(gradients[6], learning_rate)
                self.output_bias.update(gradients[7], learning_rate)

            adjusted_rate += 0.05
            # check for 97.5 % accuracy after each epoch, not after each batch
            if dataset.get_validation_accuracy() >= 0.975:
                return

