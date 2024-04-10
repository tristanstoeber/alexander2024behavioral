import unittest
from tools import *


class TestCenterDetection(unittest.TestCase):
    def test_general_case(self):
        # Test with a general case where lines intersect
        q1, q3, q2, q4 = (0, 0), (2, 2), (0, 2), (2, 0)
        self.assertEqual(detect_center_based_on_quadrant_coordinates(q1, q2, q3, q4), (1, 1))

    def test_parallel_lines(self):
        # Test with parallel lines which do not intersect
        q1, q3, q2, q4 = (0, 0), (1, 1), (0, 1), (1, 2)
        self.assertIsNone(detect_center_based_on_quadrant_coordinates(q1, q2, q3, q4))

    def test_coincident_points(self):
        # Test with coincident points (which form a line)
        q1, q3, q2, q4 = (1, 1), (1, 1), (2, 2), (3, 3)
        self.assertIsNone(detect_center_based_on_quadrant_coordinates(q1, q2, q3, q4))

    def test_collinear_points(self):
        # Test with collinear points (which do not form a proper intersection)
        q1, q3, q2, q4 = (0, 0), (1, 1), (2, 2), (3, 3)
        self.assertIsNone(detect_center_based_on_quadrant_coordinates(q1, q2, q3, q4))

class TestDetectCircleBoundaries(unittest.TestCase):

    def test_detect_circle_boundaries(self):
        # Test case with known circle boundaries
        x_coordinates = [0., 1., 2.]
        y_coordinates = [0., 1., 0.]

        center_x_expected = 1.0  # Expected center x-coordinate
        center_y_expected = 0.5  # Expected center y-coordinate
        radius_expected = 1.  # Expected radius

        center_x, center_y, radius = detect_circle_boundaries(x_coordinates, y_coordinates)
        print(center_x, center_y, radius)
        # Check if calculated values match expected values within a small tolerance (e.g., 0.001)
        self.assertAlmostEqual(center_x, center_x_expected, places=3)
        self.assertAlmostEqual(center_y, center_y_expected, places=3)
        self.assertAlmostEqual(radius, radius_expected, places=3)

if __name__ == '__main__':
    unittest.main()