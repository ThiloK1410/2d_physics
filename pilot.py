import ball
from neural_network import NeuralNetwork


class Pilot:
    def __init__(self, actor, inputs_count):
        self.actor = actor
        self.brain = NeuralNetwork([inputs_count, 3, 3, 2])

    def pilot(self, inputs):
        self.brain.calculate(inputs)
        self.actor.active_move(self.brain.outputs)
