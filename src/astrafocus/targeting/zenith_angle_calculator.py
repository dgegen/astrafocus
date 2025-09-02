import numpy as np

RAD_TO_DEG = 180 / np.pi
DEG_TO_RAD = np.pi / 180


class ZenithAngleCalculator:
    @staticmethod
    def add_zenith_angle_and_cartesian_coordinates(df, zenith):
        ZenithAngleCalculator.add_cartesian_coordinates(df)

        df.loc[:, "zenith_angle"] = (
            np.arccos(
                np.dot(
                    df.loc[:, ["x", "y", "z"]].to_numpy(),
                    np.array(
                        ZenithAngleCalculator.spherical_to_cartesian(zenith.ra.rad, zenith.dec.rad)
                    ).reshape(-1, 1),
                )
            )
            * RAD_TO_DEG
        )

    @staticmethod
    def add_cartesian_coordinates(df):
        (
            df.loc[:, "x"],
            df.loc[:, "y"],
            df.loc[:, "z"],
        ) = ZenithAngleCalculator.spherical_to_cartesian(df.ra * DEG_TO_RAD, df.dec * DEG_TO_RAD)

    @staticmethod
    def add_zenith_angle_fast(df, zenith):
        df.loc[:, "zenith_angle"] = (
            np.arccos(
                np.sin(zenith.dec.rad) * np.sin(df.dec * DEG_TO_RAD)
                + np.cos(zenith.dec.rad)
                * np.cos(df.dec * DEG_TO_RAD)
                * np.cos((zenith.ra.deg - df.ra) * DEG_TO_RAD)
            )
            * RAD_TO_DEG
        )

    @staticmethod
    def spherical_to_cartesian(phi, theta, r=1):
        x = r * np.cos(theta) * np.cos(phi)
        y = r * np.cos(theta) * np.sin(phi)
        z = r * np.sin(theta)
        return x, y, z
