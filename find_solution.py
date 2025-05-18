import numpy as np
import numpy.typing as npt
from typing import Literal
from sklearn.linear_model import Ridge
from models import get_poly_coords

def get_matrix(
        rgb_1 : npt.NDArray[np.float_],
        rgb_2 : npt.NDArray[np.float_],
        model_name : Literal["PCC2", "PCC3", "RPCC2", "RPCC3", "linear"],
        weight_decay : float = 0.0
) -> npt.NDArray[np.float_]:
    """
    This function gets polynomial or root-polynomial coorinates and finds solution matrix for model in ["PCC2", "PCC3", "RPCC2", "RPCC3"]

    Parameters
    ----------
    rgb_1 : npt.NDArray[np.float_]
        Array of n x 3 shape that includes colors in RGB space of first camera.

    rgb_2 : npt.NDArray[np.float_]
    Array of n x 3 shape that includes colors in RGB space of second camera.

    model_name : str
        Defines for which model expanded coordinates should be calculated. 

    weight_decay : float, optional
        The weight decay parameter for L2 regularization. Default is 0.0

    Returns
    -------
    matrix : npt.NDArray[np.float32]
        The solution matrix of the linear regression model of shape 3 x m, m - is number of output features (of poly or root-poly coordinates) in expanded coordinates.

    """
    assert model_name in ["PCC2", "PCC3", "RPCC2", "RPCC3", "linear"]
    
    if model_name != "linear":
        rgb_expanded = get_poly_coords(rgb_1, model_name)
    else:
        rgb_expanded = rgb_1

    A = linear_regression(rgb_expanded, rgb_2, weight_decay)

    return A


def linear_regression(
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    weight_decay: float = 0.0
) -> npt.NDArray[np.float32]:
    """
    This function finds solution matrix A of linear regression model: y = x @ A.T

    Parameters
    ----------
    x : npt.NDArray[np.float_]
        The input data matrix of shape n x m, where n - is the number of samples and m - is number of input features (of poly or root-poly coordinates).

    y : npt.NDArray[np.float_]
        The target data matrix of shape n x 3, where n - is the number of samples and 3 is number of output features (R, G, B).

    weight_decay : float, optional
        The weight decay parameter for L2 regularization. Default is 0.0

    Returns
    -------
    matrix : npt.NDArray[np.float32]
        The solution matrix of the linear regression model of shape 3 x m.
    """

    linear_model = Ridge(alpha=weight_decay, fit_intercept=False)
    solution = linear_model.fit(x, y)
    matrix = solution.coef_

    return matrix