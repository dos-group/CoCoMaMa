import numpy as np

from Hyperrectangle import Hyperrectangle

"""
This represents a node as discussed in the paper.
"""


class UcbExtendedNode:
    # The node obj is simply used to visualize the tree

    def __init__(self, parent_node, h, hyperrectangle: Hyperrectangle):
        self.parent_node = parent_node
        self.h = h
        self.hyperrectangle = hyperrectangle
        self.dimension = self.hyperrectangle.get_dimension()

    def reproduce(self):
        """
        This fun creates 2 new nodes and assigns regions (i.e. hyperrectangles) to them.
        :return: A list of the 2 new nodes.
        """
        # Pick dimension to split based on the dimension with the biggest length
        dimension_to_split = np.argsort(self.hyperrectangle.length)[0]


        # Define the length of the new Hyperrectangles
        length_modification = np.zeros(self.dimension)
        length_modification[dimension_to_split] = self.hyperrectangle.length[dimension_to_split]/2
        new_length = self.hyperrectangle.length - length_modification

        # Define the center of the new Hyperrectangles
        center_translation = length_modification / 2
        center_1 = self.hyperrectangle.center + center_translation
        center_2 = self.hyperrectangle.center - center_translation

        return [UcbNode(self, self.h + 1, Hyperrectangle(new_length, center_1)),
                UcbNode(self, self.h + 1, Hyperrectangle(new_length, center_2))]


    def contains_context(self, context):
        if self.hyperrectangle.is_pt_in_hypercube(context):
            return True
        return False

    def __str__(self):
        return str(self.h) + ": " + str(self.hyperrectangle_list)

    def __repr__(self):
        return self.__str__()
