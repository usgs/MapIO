import numpy as np
from mapio.gmt import GMTGrid


def grdcmp(x, y, rtol=1e-6, atol=0):
    """
    Compare contents of two GMT GRD files using numpy assert method.

    Args:
        x: Path to a GRD file.
        y: Another path to a GRD file.

    """
    xgrid = GMTGrid.load(x)
    xdata = xgrid.getData()
    ygrid = GMTGrid.load(y)
    ydata = ygrid.getData()
    np.testing.assert_allclose(xdata, ydata, rtol=rtol, atol=atol)
