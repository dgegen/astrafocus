from abc import ABC, abstractmethod
from typing import Union

from astrafocus.utils.typing import ImageType


class CameraInterface(ABC):
    """
    Abstract base class representing a telescope camera interface.

    Methods
    -------
    perform_exposure(texp)
        Take an observation with a specified exposure time.
    """

    @abstractmethod
    def perform_exposure(self, texp: float) -> Union[ImageType, None]:
        """
        Abstract method to take an observation with a specified exposure time.

        Parameters
        ----------
        texp : float
            Exposure time.

        Returns
        -------
        ImageType
            Resulting image.
        """
        pass

    def __repr__(self) -> str:
        return f"CameraInterface()"


class TrivialCamera(CameraInterface):
    """
    A trivial camera interface for testing purposes.
    """

    def perform_exposure(self, texp: float) -> ImageType:
        pass

    def __repr__(self) -> str:
        return f"TrivialCamera()"
