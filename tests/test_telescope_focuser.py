import unittest

from astrafocus.interface.focuser import TrivialFocuser


class TestTrivialFocuser(unittest.TestCase):
    def test_initialization(self):
        # Test initializing the focuser within the allowed range
        focuser = TrivialFocuser(0, (0, 100))
        self.assertEqual(focuser.position, 0)

        # Test initializing the focuser outside the allowed range (should raise ValueError)
        with self.assertRaises(ValueError):
            TrivialFocuser(150, (0, 100))

    def test_move_by_steps(self):
        focuser = TrivialFocuser(50, (0, 100))

        # Test moving within the allowed range
        focuser.move_by_steps(10)
        self.assertEqual(focuser.position, 60)

        # Test moving outside the allowed range (should raise ValueError)
        with self.assertRaises(ValueError):
            focuser.move_by_steps(50)

    def test_set_position(self):
        focuser = TrivialFocuser(50, (0, 100))

        # Test setting the position within the allowed range
        focuser.position = 75
        self.assertEqual(focuser.position, 75)

        # Test setting the position outside the allowed range (should raise ValueError)
        with self.assertRaises(ValueError):
            focuser.position = 150

    def test_validate_desired_position(self):
        focuser = TrivialFocuser(50, (0, 100))

        # Test valid desired position
        focuser.validate_desired_position(60)

        # Test invalid desired position (not an integer, should raise ValueError)
        with self.assertRaises(ValueError):
            focuser.validate_desired_position(55.5)

        # Test invalid desired position (outside allowed range, should raise ValueError)
        with self.assertRaises(ValueError):
            focuser.validate_desired_position(150)

    def test_is_within_range(self):
        focuser = TrivialFocuser(50, (0, 100))

        # Test within allowed range
        self.assertTrue(focuser.is_within_range(60))

        # Test outside allowed range
        self.assertFalse(focuser.is_within_range(150))

    def test_validate_allowed_range(self):
        # Test valid allowed range
        focuser = TrivialFocuser(0, (0, 100))
        focuser.validate_allowed_range()

        # Test allowed range not being a tuple, list, or numpy array (should raise ValueError)
        focuser.allowed_range = 50
        with self.assertRaises(ValueError):
            focuser.validate_allowed_range()

        # Test allowed range with non-integer boundaries (should raise ValueError)
        focuser.allowed_range = (0.5, 100)
        with self.assertRaises(ValueError):
            focuser.validate_allowed_range()

        # Test allowed range with more than 2 items (should raise ValueError)
        focuser.allowed_range = (0, 50, 100)
        with self.assertRaises(ValueError):
            focuser.validate_allowed_range()

        # Test allowed range with max_step <= min_step (should raise ValueError)
        focuser.allowed_range = (100, 0)
        with self.assertRaises(ValueError):
            focuser.validate_allowed_range()


if __name__ == "__main__":
    unittest.main()
