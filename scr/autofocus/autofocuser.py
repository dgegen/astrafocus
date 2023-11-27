from abc import ABC, abstractmethod, ABCMeta
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from autofocus.interface.telescope import TelescopeInterface
from autofocus.interface.telescope_specs import TelescopeSpecs
from autofocus.utils.logger import configure_logger
from autofocus.focus_measure_operators import (
    FocusMeasureOperator,
    AnalyticResponseFocusedMeasureOperator,
)

logger = configure_logger(stream_handler_level=10)  # logging.INFO


class AutofocuserBase(ABC):
    def __init__(
        self,
        telescope_interface: TelescopeInterface,
        focus_measure_operator: FocusMeasureOperator,
        exposure_time: float,
        search_range: Optional[Tuple[int, int]] = None,
        initial_position: Optional[int] = None,
    ):
        self.telescope_interface = telescope_interface
        self.focus_measure_operator = focus_measure_operator
        self.exposure_time = exposure_time
        self.search_range = search_range or telescope_interface.focuser.allowed_range
        self.initial_position = initial_position or telescope_interface.focuser.position

        self._focus_record = pd.DataFrame(columns=["focus_pos", "focus_measure"], dtype=np.float64)
        self.best_focus_position = None

    @property
    def focus_record(self):
        df = self._focus_record.copy()
        df["focus_pos"] = df["focus_pos"].astype(int)
        return df

    def measure_focus(self, image: np.ndarray) -> float:
        return self.focus_measure_operator(image)

    def run(self):
        try:
            self._run()
            logger.info("Successfully completed autofocusing.")
        except Exception as e:
            logger.exception(e)
            logger.warning("Error in autofocus algorithm. Resetting focuser to initial position.")
            self.reset()

    @abstractmethod
    def _run(self):
        pass

    def reset(self):
        self.telescope_interface.move_focuser_to_position(self.initial_position)

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
            f"AutofocuserBase(self.telescope_interface={self.telescope_interface!r}, "
            f"exposure_time={self.exposure_time!r} sec, "
            f"search_range={self.search_range!r}, "
            f"initial_position={self.initial_position!r})"
        )


class SweepingAutofocuser(AutofocuserBase):
    def __init__(
        self,
        telescope_interface: TelescopeInterface,
        exposure_time: float,
        focus_measure_operator,
        n_steps: Tuple[int] = (10,),
        n_exposures: Union[int, np.ndarray] = 1,
        search_range: Optional[Tuple[int, int]] = None,
        decrease_search_range=True,
        initial_position: Optional[int] = None,
    ):
        super().__init__(
            telescope_interface,
            focus_measure_operator,
            exposure_time,
            search_range,
            initial_position,
        )

        self.n_sweeps = len(n_steps)
        self.n_steps = n_steps
        self.n_exposures = (
            np.array(n_exposures, dtype=int)
            if isinstance(n_exposures, (np.ndarray, list, tuple))
            else np.full(self.n_sweeps, n_exposures, dtype=int)
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

            logger.info(
                f"Starting sweep {sweep + 1} of {self.n_sweeps}."
                f" ({search_positions[0]:6d}, {search_positions[-1]:6d}, {self.n_steps[sweep]:3d})."
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
        self.telescope_interface.move_focuser_to_position(best_focus_pos)

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
            for exposure in range(n_exposures):
                # This step must include processing such as hot pixel removal etc.
                image = self.telescope_interface.take_observation_at(
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


class AnalyticResponseAutofocuser(SweepingAutofocuser):
    """
    Autofocuser that fits a curve to the focus response curve and finds the best focus position.

    Examples
    --------
    >>> from autofocus.interface.telescope import TelescopeInterface
    >>> from autofocus.interface.simulation import get_telescope_simulation
    >>> from autofocus.star_size_focus_measure_operators import HFRStarFocusMeasure
    >>> from autofocus.autofocuser import AnalyticResponseAutofocuser
    >>> PATH_TO_FITS = 'path_to_fits'
    >>> telescope_interface = get_telescope_simulation()

    >>> np.random.seed(42)
    >>> araf = AnalyticResponseAutofocuser(
            telescope_interface=telescope_interface,
            exposure_time=3.0,
            focus_measure_operator=HFRStarFocusMeasure,
            n_steps=(30, 10),
            n_exposures=(1, 2),
            decrease_search_range=True,
            percent_to_cut=60
        )
    >>> araf.run()
    >>> araf.telescope_interface.total_time
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
        telescope_interface: TelescopeInterface,
        exposure_time: float,
        focus_measure_operator: AnalyticResponseFocusedMeasureOperator,
        percent_to_cut: float = 50.0,
        **kwargs,
    ):
        ref_image = telescope_interface.take_observation(texp=exposure_time)

        super().__init__(
            telescope_interface=telescope_interface,
            exposure_time=exposure_time,
            focus_measure_operator=focus_measure_operator(ref_image=ref_image),
            **kwargs,
        )
        self.update_search_range
        self.percent_to_cut = percent_to_cut

    def _find_best_focus_position(self, focus_pos, focus_measure) -> Tuple[int, float]:
        best_focus_pos, best_focus_val = self.fit_focus_response_curve(focus_pos, focus_measure)

        return best_focus_pos, best_focus_val

    def fit_focus_response_curve(self, focus_pos, focus_measure):
        optimal_focus_pos = self.focus_measure_operator.fit_focus_response_curve(
            focus_pos, focus_measure
        )
        optimal_focus_pos = int(np.round(optimal_focus_pos))
        best_focus_val = self.focus_measure_operator.get_focus_response_curve_fit(optimal_focus_pos)

        return optimal_focus_pos, best_focus_val

    def get_focus_response_curve_fit(self, focus_pos):
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
