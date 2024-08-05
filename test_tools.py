import unittest
import tools

class TestDetectCircleBoundaries(unittest.TestCase):

    def test_detect_circle_boundaries(self):
        # Test case with known circle boundaries
        x_coordinates = [0., 1., 2.]
        y_coordinates = [0., 1., 0.]

        center_x_expected = 1.0  # Expected center x-coordinate
        center_y_expected = 0.5  # Expected center y-coordinate
        radius_expected = 1.  # Expected radius

        center_x, center_y, radius = tools.detect_circle_boundaries(x_coordinates, y_coordinates)
        print(center_x, center_y, radius)
        # Check if calculated values match expected values within a small tolerance (e.g., 0.001)
        self.assertAlmostEqual(center_x, center_x_expected, places=3)
        self.assertAlmostEqual(center_y, center_y_expected, places=3)
        self.assertAlmostEqual(radius, radius_expected, places=3)

if __name__ == '__main__':
    unittest.main()