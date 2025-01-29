import numpy as np
from numpy.typing import NDArray


class LandExtrap:
    """
    This class is for implementing sea-over-land procedure,
    to extrapolate currents over land.
    """

    iterations = 5

    @staticmethod
    def extrap3d(datasource: NDArray, method: str = "average") -> NDArray:
        """
        Extrapolation of 3-D fields over land points.

        Supported methods are 'average', 'gradient'
        """
        match method:
            case "average":
                datasource = LandExtrap.__apply_average_sea_over_land(datasource)
            case "gradient":
                datasource = datasource # method will be applied within model core
            case _:
                raise KeyError(
                    f"Sea Over Land method '{method}' is not dupported. \
                                 Supported methods are 'average', 'gradient'."
                )
        return datasource

    @staticmethod
    def __apply_average_sea_over_land(datasource: NDArray) -> NDArray:
        """
        Apply Extrapolation based on average method.
        """
        try:
            nx, ny, nz = datasource.shape
            for k in range(nz):
                datasource[:, :, k] = LandExtrap.__average_sea_over_land(
                    datasource[:, :, k]
                )
        except ValueError:
            datasource = LandExtrap.__average_sea_over_land(datasource)
        return datasource

    @staticmethod
    def __average_sea_over_land(datasource: NDArray) -> NDArray:
        """
        Extrapolation of 2-D fields over land points.
        Points are extrapolated averaging the neighbouring ones.

        Parameters:
        datasource (numpy.ndarray): 2-D array to be processed.

        Returns:
        numpy.ndarray: Processed 2-D array.
        """
        nx, ny = datasource.shape

        for _ in range(LandExtrap.iterations):
            carpet = datasource.copy()
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    if np.isnan(datasource[i, j]):
                        datan = 0.0
                        jcn = 0
                        im1, ip1 = i - 1, i + 1
                        jm1, jp1 = j - 1, j + 1
                        for jj in range(jm1, jp1 + 1):
                            for ii in range(im1, ip1 + 1):
                                if not np.isnan(datasource[ii, jj]):
                                    datan += datasource[ii, jj]
                                    jcn += 1
                        if jcn >= 2:
                            carpet[i, j] = datan / jcn
            datasource = carpet
        return datasource


if __name__ == "__main__":
    datasource = np.random.rand(3, 3, 4)
    datasource[1, 1, :] = np.nan  # Introducing NaNs for testing
    datasource = LandExtrap.extrap3d(datasource, method="average")
    test = np.allclose(datasource[1, 1, :], np.mean(datasource, axis=(0, 1)))
    assert np.all(test), "AVERAGE METHOD not working"
    datasource = np.random.rand(10, 10, 4)
    datasource[4, 4, :] = np.nan  # Introducing NaNs for testing
    print(datasource)
    datasource = LandExtrap.extrap3d(datasource, method="gradient")
    assert np.all(np.isnan(datasource[4, 4, :])), "GRADIENT METHOD not working"
