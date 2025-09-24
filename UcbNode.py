import numpy as np

from Hyperrectangle import Hyperrectangle

"""
This represents a node as discussed in the paper.
"""


class UcbNode:
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
        # Get all dimensions that have the maximum length
        max_length = np.max(self.hyperrectangle.length)
        max_dimensions = np.where(self.hyperrectangle.length == max_length)[0]
        # Randomly select one of the dimensions with maximum length
        dimension_to_split = np.random.choice(max_dimensions)

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
    
    def reproduce_informed(self, cov_context_reward_dict, avg_context_dict):
        """
        This fun creates 2 new nodes and assigns regions (i.e. hyperrectangles) to them. The node is split on the dimension with the highest absolute context-reward covariance. The split is made on the mean in that dimension,
        :return: A list of the 2 new nodes.
        """
        # Sort by absolute covariance values to consider both positive and negative correlations
        dimension_to_split = np.argsort(np.abs(cov_context_reward_dict))[0]

        old_center = self.hyperrectangle.center[dimension_to_split]
        old_length = self.hyperrectangle.length[dimension_to_split]
        split_location = avg_context_dict[dimension_to_split]

        length_1 = abs(split_location - (old_center - old_length/2))
        length_2 = abs((old_center + old_length/2) - split_location)

        new_center_1 = split_location - length_1 / 2
        new_center_2 = split_location + length_2 / 2

        length_vector_1 = np.copy(self.hyperrectangle.length)
        length_vector_1[dimension_to_split] = length_1
        center_vector_1 = np.copy(self.hyperrectangle.center)
        center_vector_1[dimension_to_split] = new_center_1

        length_vector_2 = np.copy(self.hyperrectangle.length)
        length_vector_2[dimension_to_split] = length_2
        center_vector_2 = np.copy(self.hyperrectangle.center)
        center_vector_2[dimension_to_split] = new_center_2
        
        return [UcbNode(self, self.h + 1, Hyperrectangle(length_vector_1, center_vector_1)),
                UcbNode(self, self.h + 1, Hyperrectangle(length_vector_2, center_vector_2))]



    def contains_context(self, context):
        if self.hyperrectangle.is_pt_in_hypercube(context):
            return True
        return False

    def __str__(self):
        return str(self.h) + ": " + str(self.hyperrectangle_list)

    def __repr__(self):
        return self.__str__()
