import numpy as np

try:
    from numba import njit  # type: ignore
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

if _NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _numba_point_in_box(point, center, half_length, eps):
        for i in range(point.shape[0]):
            if abs(point[i] - center[i]) > half_length[i] + eps:
                return False
        return True
else:
    def _numba_point_in_box(point, center, half_length, eps):
        for i in range(point.shape[0]):
            if abs(point[i] - center[i]) > half_length[i] + eps:
                return False
        return True


"""
Represents a hyperrectangle context region.
"""


class Hyperrectangle:
    """
    Represents a hyperrectangle (also known as a box or n-dimensional rectangle) in an n-dimensional space.

    Attributes:
        length (vector): The side lengths of the hyperrectangle in each dimension. Should be a vector (e.g., list, numpy array).
        center (vector): The coordinates of the center of the hyperrectangle. Should be a vector (e.g., list, numpy array).
    """

    def __init__(self, length, center):
        """
        Initializes a Hyperrectangle.

        Args:
            length (vector): The side lengths of the hyperrectangle in each dimension.
            center (vector): The coordinates of the center of the hyperrectangle.
        """
        # Ensure contiguous numeric arrays and precompute half-lengths
        center_arr = np.asarray(center)
        length_arr = np.asarray(length)
        dtype = center_arr.dtype if center_arr.dtype.kind == 'f' else np.float64
        self.center = np.ascontiguousarray(center_arr, dtype=dtype)
        self.length = np.ascontiguousarray(length_arr, dtype=dtype)
        self.half_length = np.ascontiguousarray(self.length * 0.5, dtype=dtype)
        self._eps = dtype.type(1e-10)

    def is_pt_in_hypercube(self, point):
        """
        Checks if a given point is inside the hyperrectangle.

        Args:
            point (vector): The coordinates of the point to check.

        Returns:
            bool: True if the point is inside the hyperrectangle, False otherwise.
        """
        # Fast path using precomputed arrays and optional numba kernel
        p = np.asarray(point, dtype=self.center.dtype)
        return _numba_point_in_box(p, self.center, self.half_length, self._eps)

    def get_dimension(self):
        """
        Returns the number of dimensions of the hyperrectangle.

        Returns:
            int: The number of dimensions.
        """
        return len(self.center)

    def __str__(self):
        """
        Returns a string representation of the hyperrectangle.

        Returns:
            str: String representation.
        """
        if len(self.center) == 1:  # for testing purposes
            return "[" + str(self.center[0] - self.length / 2) + " - " + str(self.center[0] + self.length / 2)
        return "center: " + str(self.center) + " length: " + str(self.length)

    def __repr__(self):
        """
        Returns a string representation of the hyperrectangle (for debugging).

        Returns:
            str: String representation.
        """
        return self.__str__()
