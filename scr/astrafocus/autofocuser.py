import os
import time
from abc import ABC, ABCMeta, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from astrafocus.focus_measure_operators import (
    AnalyticResponseFocusedMeasureOperator,
    FocusMeasureOperator,
)
from astrafocus.extremum_estimators import RobustExtremumEstimator, LOWESSExtremumEstimator
from astrafocus.interface.device_manager import AutofocusDeviceManager
from astrafocus.utils.logger import configure_logger


logger = configure_logger(stream_handler_level=10)  # logging.INFO


class AutofocuserBase(ABC):
    """
    Abstract base class for autofocusing algorithms.

    Parameters
    ----------
    autofocus_device_manager : AutofocusDeviceManager
        Interface to control the camera and its focuser.
    focus_measure_operator : FocusMeasureOperator
        Operator to measure the focus of images.
    exposure_time : float
        Exposure time for image acquisition.
    search_range : Optional[Tuple[int, int]], optional
        Range of focus positions to search for the best focus (default is None,
        using the telescope's allowed range).
    initial_position : Optional[int], optional
        Initial focus position for the autofocus algorithm (default is None,
        using the telescope's current position).
    keep_images : bool, optional
        Whether to keep images for additional analysis (default is False).
    secondary_focus_measure_operators : Optional[dict], optional
        Dictionary of additional focus measure operators for image analysis
        (default is an empty dictionary).
    save_path : Optional[str], optional
        Path to save focus record to (default is None). If None, the focus record is not saved.
        If the path ends with '.csv', the focus record is saved with that name. Otherwise, the
        focus record is saved as a csv file with a timestamp in the specified directory.

    Attributes
    ----------
    focus_record : pd.DataFrame
        DataFrame containing focus positions and corresponding focus measures.
    best_focus_position : int or None
        Best focus position determined by the autofocus algorithm.
    _image_record : list
        List to store images if 'keep_images' is True.

    Methods
    -------
    measure_focus(image: np.ndarray) -> float:
        Measure the focus of a given image using the specified focus measure operator.
    run():
        Execute the autofocus algorithm. Handles exceptions and resets the focuser on failure.
    _run():
        Abstract method to be implemented by subclasses for the actual autofocus algorithm.
    reset():
        Reset the focuser to the initial position.
    get_focus_record() -> Tuple[np.ndarray, np.ndarray]:
        Retrieve the focus record as sorted arrays of focus positions and corresponding measures.

    Examples
    --------
    # Instantiate an AutofocuserBase instance
    >>> autofocus_instance = AutofocuserBase(
    ...     autofocus_device_manager, focus_measure_operator, exposure_time
    ... )

    # Run the autofocus algorithm
    >>> autofocus_instance.run()
    """

    def __init__(
        self,
        autofocus_device_manager: AutofocusDeviceManager,
        focus_measure_operator: FocusMeasureOperator,
        exposure_time: float,
        search_range: Optional[Tuple[int, int]] = None,
        initial_position: Optional[int] = None,
        keep_images: bool = False,
        secondary_focus_measure_operators: Optional[dict] = None,
        save_path: Optional[str] = None,
    ):
        self.autofocus_device_manager = autofocus_device_manager
        self.focus_measure_operator = focus_measure_operator
        self.exposure_time = exposure_time
        self.search_range = search_range or autofocus_device_manager.focuser.allowed_range
        self.initial_position = initial_position or autofocus_device_manager.focuser.position

        self._focus_record = pd.DataFrame(columns=["focus_pos", "focus_measure"], dtype=np.float64)
        self.best_focus_position = None

        self.keep_images = keep_images
        self.secondary_focus_measure_operators = secondary_focus_measure_operators or {}
        self._image_record = []
        self.save_path = save_path

    @property
    def focus_record(self):
        df = self._focus_record.copy()
        df["focus_pos"] = df["focus_pos"].astype(int)

        if self.keep_images:
            for name, fm in self.secondary_focus_measure_operators.items():
                df[name] = np.array([fm.measure_focus(image) for image in self._image_record])

        return df

    def measure_focus(self, image: np.ndarray) -> float:
        if self.keep_images:
            self._image_record.append(image)
        return self.focus_measure_operator(image)

    def run(self) -> bool:
        try:
            self._run()
            logger.info("Successfully completed autofocusing.")
            success = True
            self.save_focus_record()
        except Exception as e:
            logger.exception(e)
            logger.warning("Error in autofocus algorithm. Resetting focuser to initial position.")
            success = False
            self.reset()

        return success

    @abstractmethod
    def _run(self):
        pass

    def reset(self):
        self.autofocus_device_manager.move_focuser_to_position(self.initial_position)

    def save_focus_record(self):
        if not isinstance(self.save_path, str):
            if self.save_path is not None:
                logger.warning("Error saving focus record to csv. Invalid save path.")
            return None

        try:
            if self.save_path.endswith(".csv"):
                save_path = self.save_path
            else:
                timestr = time.strftime('%Y-%m-%dT%H:%M:%S')
                save_path = os.path.join(self.save_path, f"{timestr}_focus_record.csv")

            self.focus_record.to_csv(save_path, index=False)
        except Exception as e:
            logger.exception(e)
            logger.warning("Error saving focus record to csv.")

    def get_focus_record(self):
        if self._focus_record.size == 0:
            raise ValueError("Focus record is empty. Run the autofocus algorithm first.")

        focus_pos = self._focus_record.focus_pos[~np.isnan(self._focus_record.focus_pos)].to_numpy(
            int
        )
        focus_measure = self._focus_record.focus_measure[
            ~np.isnan(self._focus_record.focus_measure)
        ].to_numpy()
        sort_ind = np.argsort(focus_pos)

        return focus_pos[sort_ind], focus_measure[sort_ind]

    def __repr__(self) -> str:
        return (
            f"AutofocuserBase(self.autofocus_device_manager={self.autofocus_device_manager!r}, "
            f"exposure_time={self.exposure_time!r} sec, "
            f"search_range={self.search_range!r}, "
            f"initial_position={self.initial_position!r})"
        )


class SweepingAutofocuser(AutofocuserBase):
    """
    Autofocuser implementation using a sweeping algorithm.

    Parameters
    ----------
    autofocus_device_manager : AutofocusDeviceManager
        Interface to control the camera and its focuser.
    exposure_time : float
        Exposure time for image acquisition.
    focus_measure_operator : FocusMeasureOperator
        Operator to measure the focus of images.
    n_steps : Tuple[int], optional
        Number of steps for each sweep (default is (10,)).
        The length of this tuple determines the number of sweeps. The entries specify the number of
        steps for each sweep.
    n_exposures : Union[int, np.ndarray], optional
        Number of exposures at each focus position or an array specifying exposures for each sweep.
        If an integer is given, the same number of exposures is used for each sweep (default is 1).
        If an array is given, the length of the array must match the number of sweeps.
        (default is 1).
    search_range : Optional[Tuple[int, int]], optional
        Range of focus positions to search for the best focus (default is None,
        using the telescope's allowed range).
    decrease_search_range : bool, optional
        Whether to decrease the search range after each sweep (default is True).
    initial_position : Optional[int], optional
        Initial focus position for the autofocus algorithm (default is None,
        using the telescope's current position).
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    n_sweeps : int
        Number of sweeps to perform.
    n_steps : Tuple[int]
        Number of steps for each sweep.
    n_exposures : np.ndarray
        Number of exposures at each focus position.
    decrease_search_range : bool
        Whether to decrease the search range after each sweep.

    Methods
    -------
    _run():
        Execute the sweeping autofocus algorithm.
    get_initial_direction(min_focus_pos, max_focus_pos) -> int:
        Determine the initial direction of the sweep.
    find_best_focus_position():
        Find and set the best focus position based on the recorded focus measures.
    _find_best_focus_position(focus_pos, focus_measure) -> Tuple[int, float]:
        Abstract method to be implemented by subclasses for finding the best focus position.
    _run_sweep(search_positions, n_exposures):
        Perform a single sweep across the specified focus positions.
    update_search_range(min_focus_pos, max_focus_pos) -> Tuple[int, int]:
        Update the search range after each sweep.
    integer_linspace(min_focus_pos, max_focus_pos, n_steps) -> np.ndarray:
        Generate integer-spaced values within the specified range.

    Examples
    --------
    # Instantiate a SweepingAutofocuser instance
    >>> sweeping_autofocuser = SweepingAutofocuser(autofocus_device_manager, exposure_time, focus_measure_operator)

    # Run the sweeping autofocus algorithm
    >>> sweeping_autofocuser.run()
    """

    def __init__(
        self,
        autofocus_device_manager: AutofocusDeviceManager,
        exposure_time: float,
        focus_measure_operator,
        n_steps: Tuple[int] = (10,),
        n_exposures: Union[int, np.ndarray] = 1,
        search_range: Optional[Tuple[int, int]] = None,
        decrease_search_range=True,
        initial_position: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            autofocus_device_manager,
            focus_measure_operator,
            exposure_time,
            search_range,
            initial_position,
            **kwargs,
        )

        self.n_sweeps = len(n_steps)
        self.n_steps = n_steps
        self.n_exposures = (
            np.array(n_exposures, dtype=int)
            if isinstance(n_exposures, (np.ndarray, list, tuple))
            else np.full(self.n_sweeps, n_exposures, dtype=int)
        )
        if len(self.n_exposures) != self.n_sweeps:
            raise ValueError(
                f"Length of n_exposures ({len(self.n_exposures)}) must match length of n_steps "
                f"({self.n_sweeps})."
            )

        self._focus_record = pd.DataFrame(
            np.full((np.sum(np.array(n_steps) * self.n_exposures), 2), np.nan),
            columns=self._focus_record.columns,
            dtype=np.float64,
        )
        self.decrease_search_range = decrease_search_range

    def _run(self):
        min_focus_pos, max_focus_pos = self.search_range
        initial_direction = self.get_initial_direction(min_focus_pos, max_focus_pos)

        for sweep in range(self.n_sweeps):
            search_positions = self.integer_linspace(
                min_focus_pos, max_focus_pos, self.n_steps[sweep]
            )

            if sweep % 2 == initial_direction:
                search_positions = np.flip(search_positions)  # Reverse order

            if not self.autofocus_device_manager.check_conditions():
                raise ValueError("Observation conditions are not good enough to take exposures.")

            logger.info(
                f"Starting sweep {sweep + 1} of {self.n_sweeps}."
                + f" ({np.min(search_positions)}, {np.max(search_positions)}, "
                + f"{self.n_steps[sweep]}"
                + (", reversed" if sweep % 2 == initial_direction else "")
                + ")."
            )
            self._run_sweep(search_positions, self.n_exposures[sweep])

            if self.decrease_search_range:
                min_focus_pos, max_focus_pos = self.update_search_range(
                    min_focus_pos, max_focus_pos
                )

        self.find_best_focus_position()

    def get_initial_direction(self, min_focus_pos, max_focus_pos):
        """Move upward if initial position is closer to min_focus_pos than max_focus_pos."""
        initial_direction = (
            1
            if np.abs(self.initial_position - min_focus_pos)
            < np.abs(self.initial_position - max_focus_pos)
            else 0
        )
        return initial_direction

    def find_best_focus_position(self):
        focus_pos, focus_measure = self.get_focus_record()

        best_focus_pos, best_focus_measure = self._find_best_focus_position(
            focus_pos, focus_measure
        )

        self.best_focus_position = best_focus_pos
        self.autofocus_device_manager.move_focuser_to_position(best_focus_pos)

        logger.info(
            f"Best focus position: {best_focus_pos} with focus measure value: {best_focus_measure:8.3e}"
        )

    @abstractmethod
    def _find_best_focus_position(self, focus_pos, focus_measure) -> Tuple[int, float]:
        # min_ind = np.argmin(focus_measure)
        # best_focus_pos, best_focus_val = focus_pos[min_ind], focus_measure[min_ind]
        pass

    def _run_sweep(self, search_positions, n_exposures):
        start_index = np.where(np.isnan(self._focus_record.iloc[:, 0]))[0][0]

        for ind, focus_position in enumerate(search_positions):
            if not self.autofocus_device_manager.check_conditions():
                raise ValueError("Observation conditions are not good enough to take exposures.")
            for exposure in range(n_exposures):
                if not self.autofocus_device_manager.check_conditions():
                    raise ValueError(
                        "Observation conditions are not good enough to take exposures."
                    )

                # This step should include processing such as hot pixel removal etc.
                image = self.autofocus_device_manager.perform_exposure_at(
                    focus_position=focus_position, texp=self.exposure_time
                )

                fm_value = self.measure_focus(image)
                logger.debug(
                    f"Obtained measure value: {fm_value:8.3e} at focus position: {focus_position}"
                )

                # Save to record
                df_index = start_index + ind * n_exposures + exposure
                self._focus_record.loc[df_index, "focus_pos"] = focus_position
                self._focus_record.loc[df_index, "focus_measure"] = fm_value

            mean_fm_value = np.mean(
                self._focus_record.loc[
                    start_index + ind * n_exposures : start_index + ind * (n_exposures + 1),
                    "focus_measure",
                ]
            )
            if mean_fm_value < 1e3:
                logger.info(f"{focus_position:6d} | {mean_fm_value:8.3f}")
            else:
                logger.info(f"{focus_position:6d} | {mean_fm_value:8.3e}")

    def update_search_range(self, min_focus_pos, max_focus_pos):
        return min_focus_pos, max_focus_pos

    @staticmethod
    def integer_linspace(min_focus_pos, max_focus_pos, n_steps):
        """
        Notes
        -----
        Search positions can be redundant
        >>> integer_linspace(0, 1, 4)
        array([0, 0, 1, 1])
        """
        search_positions = np.array(
            np.round(np.linspace(min_focus_pos, max_focus_pos, n_steps)), dtype=int
        )
        return search_positions


class NonParametricResponseAutofocuser(SweepingAutofocuser):
    def __init__(
        self,
        autofocus_device_manager,
        exposure_time,
        focus_measure_operator,
        extremum_estimator: RobustExtremumEstimator = LOWESSExtremumEstimator(frac=0.5, it=3),
        **kwargs,
    ):
        super().__init__(autofocus_device_manager, exposure_time, focus_measure_operator, **kwargs)
        self.extremum_estimator = extremum_estimator

    def _find_best_focus_position(
        self, focus_pos: np.ndarray, focus_measure: np.ndarray
    ) -> Tuple[int, float]:
        focus_pos, focus_measure = self.get_focus_record()

        focus_pos_sorted, focus_measure_sorted = self.extremum_estimator.sort(
            focus_pos, focus_measure
        )

        # Use RobustExtremumEstimator to find the best focus position
        if self.focus_measure_operator.smaller_is_better:
            best_focus_pos, best_focus_measure_value = self.extremum_estimator.argmin(
                focus_pos, focus_measure, return_value=True
            )
        else:
            best_focus_pos, best_focus_measure_value = self.extremum_estimator.argmax(
                focus_pos, focus_measure, return_value=True
            )

        best_focus_pos = int(np.round(best_focus_pos))
        best_focus_measure_value = float(best_focus_measure_value)

        return best_focus_pos, best_focus_measure_value

    def __repr__(self) -> str:
        return (
            f"NonParametricAutofocuser(self.autofocus_device_manager={self.autofocus_device_manager!r}, "
            f"exposure_time={self.exposure_time!r} sec, "
            f"search_range={self.search_range!r}, "
            f"initial_position={self.initial_position!r}, "
            f"robust_estimator={self.robust_estimator!r})"
        )


class AnalyticResponseAutofocuser(SweepingAutofocuser):
    """
    Autofocuser that fits a curve to the focus response curve and finds the best focus position.

    Parameters
    ----------
    autofocus_device_manager : AutofocusDeviceManager
        Interface to control the telescope and its focuser.
    exposure_time : float
        Exposure time for image acquisition.
    focus_measure_operator : AnalyticResponseFocusedMeasureOperator
        Operator to measure the focus of images using an analytic response curve.
    percent_to_cut : float, optional
        Percentage of worst-performing focus positions to exclude when updating the search range
        (default is 50.0).
    **kwargs
        Additional keyword arguments.

    Examples
    --------
    >>> from astrafocus.interface.device_manager import AutofocusDeviceManager
    >>> from astrafocus.interface.simulation import ObservationBasedDeviceSimulator
    >>> from astrafocus.star_size_focus_measure_operators import HFRStarFocusMeasure
    >>> from astrafocus.autofocuser import AnalyticResponseAutofocuser
    >>> PATH_TO_FITS = 'path_to_fits'
    >>> autofocus_device_manager = ObservationBasedDeviceSimulator(fits_path=PATH_TO_FITS)

    >>> np.random.seed(42)
    >>> araf = AnalyticResponseAutofocuser(
            autofocus_device_manager=autofocus_device_manager,
            exposure_time=3.0,
            focus_measure_operator=HFRStarFocusMeasure,
            n_steps=(30, 10),
            n_exposures=(1, 2),
            decrease_search_range=True,
            percent_to_cut=60
        )
    >>> araf.run()
    >>> araf.autofocus_device_manager.total_time
    >>> araf.focus_record

    >>> import matplotlib.pyplot as plt
    >>> plt.scatter(
        araf.focus_record.focus_pos, araf.focus_record.focus_measure, ls='', marker='.'
    )
    >>> sampled_pos = np.linspace(*araf.search_range, 100)
    >>> sampled_responses = araf.get_focus_response_curve_fit(sampled_pos)
    >>> plt.plot(sampled_pos, sampled_responses)
    >>> plt.axvline(araf.best_focus_position)
    >>> plt.show()

    """

    def __init__(
        self,
        autofocus_device_manager: AutofocusDeviceManager,
        exposure_time: float,
        focus_measure_operator: AnalyticResponseFocusedMeasureOperator,
        percent_to_cut: float = 50.0,
        **kwargs,
    ):
        ref_image = autofocus_device_manager.camera.perform_exposure(texp=exposure_time)

        super().__init__(
            autofocus_device_manager=autofocus_device_manager,
            exposure_time=exposure_time,
            focus_measure_operator=focus_measure_operator(ref_image=ref_image),
            **kwargs,
        )
        self.update_search_range
        self.percent_to_cut = percent_to_cut

    def _find_best_focus_position(
        self, focus_pos: np.ndarray, focus_measure: np.ndarray
    ) -> Tuple[int, float]:
        return self.fit_focus_response_curve(focus_pos, focus_measure)

    def fit_focus_response_curve(self, focus_pos: np.ndarray, focus_measure: np.ndarray):
        optimal_focus_pos = self.focus_measure_operator.fit_focus_response_curve(
            focus_pos, focus_measure
        )
        optimal_focus_pos = int(np.round(optimal_focus_pos))
        best_focus_val = self.focus_measure_operator.get_focus_response_curve_fit(optimal_focus_pos)

        return optimal_focus_pos, best_focus_val

    def get_focus_response_curve_fit(self, focus_pos: int):
        focus_response_curve_fit = self.focus_measure_operator.get_focus_response_curve_fit(
            focus_pos
        )
        return focus_response_curve_fit

    def update_search_range(self, min_focus_pos, max_focus_pos) -> Tuple[int, int]:
        """
        Update the search range for optimal focus position based on focus response curve.

        Notes
        -----
        This function updates the search range for the optimal focus position based on the
        focus response curve. It identifies the worst-performing positions in the current
        interval and adjusts the interval accordingly.
        """
        # Get focus data and fit focus response curve parameters
        focus_pos, focus_measure = self.get_focus_record()
        _ = self.focus_measure_operator.fit_focus_response_curve(focus_pos, focus_measure)

        # Generate focus response curve
        sampled_pos = np.linspace(min_focus_pos, max_focus_pos, 100)
        sampled_responses = self.get_focus_response_curve_fit(sampled_pos)

        # Find threshold to exclude worst values
        threshold = np.percentile(sampled_responses, self.percent_to_cut)

        # Identify indices with responses below threshold
        below_threshold_indices = np.where(sampled_responses < threshold)[0]

        # Update focus search interval based on below-threshold positions
        new_min_focus_pos = np.maximum(min_focus_pos, np.min(sampled_pos[below_threshold_indices]))
        new_max_focus_pos = np.minimum(max_focus_pos, np.max(sampled_pos[below_threshold_indices]))
        new_min_focus_pos = int(np.floor(new_min_focus_pos))
        new_max_focus_pos = int(np.floor(new_max_focus_pos))

        logger.info(
            f"Updating search range from ({min_focus_pos}, {max_focus_pos}) to "
            f"({new_min_focus_pos}, {new_max_focus_pos})."
        )
        return new_min_focus_pos, new_max_focus_pos
