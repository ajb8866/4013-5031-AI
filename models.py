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
        # Sets the batch size to 24
        # Sets the hidden layer size to 350
        # Set the number of labels to 10
        # This numbers allow for the 97% requirement for the autograder
        "*** YOUR CODE HERE ***"
        self.batch_size = 10
        self.hidden_layer_size = 350
        self.num_labels = 10

        # Hidden Layer 1
        self.w_1 = nn.Parameter(784, self.hidden_layer_size)
        self.b_1 = nn.Parameter(1, self.hidden_layer_size)

        # Hidden Layer 2
        self.w_2 = nn.Parameter(self.hidden_layer_size, self.hidden_layer_size)
        self.b_2 = nn.Parameter(1, self.hidden_layer_size)

        # Hidden Layer 3
        self.w_3 = nn.Parameter(self.hidden_layer_size, self.hidden_layer_size)
        self.b_3 = nn.Parameter(1, self.hidden_layer_size)

        # Output of hidden layers
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
        # Hidden Layer 1
        t_1 = nn.Linear(x, self.w_1)
        t_b_1 = nn.AddBias(t_1, self.b_1)
        hidden_layer_1 = nn.ReLU(t_b_1)

        # Hidden Layer 2
        t_2 = nn.Linear(hidden_layer_1, self.w_2)
        t_b_2 = nn.AddBias(t_2, self.b_2)
        hidden_layer_2 = nn.ReLU(t_b_2)

        # Hidden Layer 3
        t_3 = nn.Linear(hidden_layer_2, self.w_3)
        t_b_3 = nn.AddBias(t_3, self.b_3)
        hidden_layer_3 = nn.ReLU(t_b_3)

        # Output Vector
        t_out = nn.Linear(hidden_layer_3, self.output_wt)
        return nn.AddBias(t_out, self.output_bias)

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
        # the estimated value of a response variable in a linear regression model
        y_hats = self.run(x)
        return nn.SoftmaxLoss(y_hats, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # a rate statistically adjusted to remove the effect of a variable in this cas to help
        # balance out the learning rate
        adjusted_rate = -0.12
        while True:

            # sets up the gradient based on the given inputs when creating a DigiClass object
            # then changes the learning rate accordingly
            for row, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(row, y)
                gradients_parameters = ([self.w_1, self.w_2, self.w_3, self.output_wt,
                                         self.b_1, self.b_2, self.b_3, self.output_bias])
                gradients = nn.gradients(loss, gradients_parameters)
                learning_rate = min(-0.005, adjusted_rate)

                # updates the gradients with the learning rates
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
