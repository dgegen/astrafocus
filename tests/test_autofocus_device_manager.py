import unittest
from unittest.mock import MagicMock

from astrafocus.interface.camera import TrivialCamera
from astrafocus.interface.device_manager import AutofocusDeviceManager
from astrafocus.interface.focuser import TrivialFocuser
from astrafocus.interface.telescope import TrivialTelescope


class TestAutofocusDeviceManager(unittest.TestCase):
    def setUp(self):
        self.camera = TrivialCamera()
        self.focuser = TrivialFocuser(current_position=0, allowed_range=(0, 1000))
        self.telescope = TrivialTelescope()
        self.device_manager = AutofocusDeviceManager(
            camera=self.camera, focuser=self.focuser, telescope=self.telescope
        )

    def test_initialization(self):
        self.assertEqual(self.device_manager.camera, self.camera)
        self.assertEqual(self.device_manager.focuser, self.focuser)
        self.assertEqual(self.device_manager.telescope, self.telescope)

    def test_perform_exposure_at(self):
        focus_position = 100
        exposure_time = 2.0
        expected_image = "mocked_image"

        # Mocking the camera perform_exposure method
        self.device_manager.camera.perform_exposure = MagicMock(return_value=expected_image)

        result = self.device_manager.perform_exposure_at(focus_position, exposure_time)

        self.assertEqual(result, expected_image)
        self.assertEqual(self.device_manager.focuser.position, focus_position)
        self.device_manager.camera.perform_exposure.assert_called_once_with(texp=exposure_time)

    def test_move_focuser_to_position(self):
        desired_position = 500
        self.device_manager.move_focuser_to_position(desired_position)
        self.assertEqual(self.device_manager.focuser.position, desired_position)


if __name__ == "__main__":
    unittest.main()
