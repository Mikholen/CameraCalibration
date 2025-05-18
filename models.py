import numpy as np
import numpy.typing as npt


def get_poly_coords(
    rgb : npt.NDArray[np.float_],
    model_name : str    
) -> npt.NDArray[np.float_]:
    """
    This function calculates polynomial coordinates for PCC2, PCC3, RPCC2 or RPCC3 fromn input rgb.

    For PCC2 : P_2 = ( r, g, b, r^2, g^2, b^2, rg, gb, rb ).T
    For PCC3 : P_3 = ( r, g, b, r^2, g^2, b^2, rg, gb, rb, r^3, g^3, b^3, r^2g, r^2b, g^2r, g^2b, b^2r, b^2g, rgb ).T

    For RPCC2 : RP_2 = ( r, g, b, sqrt(rg), sqrt(gb), sqrt(rb) ).T
    For RPCC3 : RP_3 = ( r, g, b, sqrt(rg), sqrt(gb), sqrt(rb), (r^2g) ** 1/3, (r^2b) ** 1/3, (g^2r) ** 1/3, (g^2b) ** 1/3, (b^2r) ** 1/3, (b^2g) ** 1/3, (rgb) ** 1/3 ).T

    Parameters
    ----------
    rgb : npt.NDArray[np.float_]
        Array of n x 3 shape that includes colors in RGB space.

    model_name : str
        Defines for which model coordinates should be calculated.

    Returns
    -------
    rgb_expanded : npt.NDArray[np.float_]
        Array of n x m shape that includes expanded rgb coordinates, m - is number of output features (of poly or root-poly coordinates).
    
    """
    
    assert model_name in ["PCC2", "PCC3", "RPCC2", "RPCC3"]

    if model_name[0] == 'R':
        if model_name[-1] == '3':
            rgb_expanded = np.concatenate([
                    rgb,
                    np.sqrt(rgb[:, :1] * rgb[:, 1:2]),
                    np.sqrt(rgb[:, 1:2] * rgb[:, 2:]),
                    np.sqrt(rgb[:, :1] * rgb[:, 2:]),
                    (rgb[:, :1]**2 * rgb[:, 1:2]) ** (1 / 3),
                    (rgb[:, :1]**2 * rgb[:, 2:]) ** (1 / 3),
                    (rgb[:, 1:2]**2 * rgb[:, :1]) ** (1 / 3),
                    (rgb[:, 1:2]**2 * rgb[:, 2:]) ** (1 / 3),
                    (rgb[:, 2:]**2 * rgb[:, :1]) ** (1 / 3),
                    (rgb[:, 2:]**2 * rgb[:, 1:2]) ** (1 / 3),
                    (rgb[:, :1] * rgb[:, 1:2] * rgb[:, 2:]) ** (1 / 3)
                ], axis=1)

        elif model_name[-1] == '2':
            rgb_expanded = np.concatenate([
                    rgb,
                    np.sqrt(rgb[:, :1] * rgb[:, 1:2]),
                    np.sqrt(rgb[:, 1:2] * rgb[:, 2:]),
                    np.sqrt(rgb[:, :1] * rgb[:, 2:])
                ], axis=1)

    else:
        if model_name[-1] == '3':
            rgb_expanded = np.concatenate([
                    rgb,
                    rgb**2,
                    rgb[:, :1] * rgb[:, 1:2],
                    rgb[:, 1:2] * rgb[:, 2:],
                    rgb[:, :1] * rgb[:, 2:],
                    rgb**3,
                    rgb[:, :1]**2 * rgb[:, 1:2],
                    rgb[:, :1]**2 * rgb[:, 2:],
                    rgb[:, 1:2]**2 * rgb[:, :1],
                    rgb[:, 1:2]**2 * rgb[:, 2:],
                    rgb[:, 2:]**2 * rgb[:, :1],
                    rgb[:, 2:]**2 * rgb[:, 1:2],
                    rgb[:, :1] * rgb[:, 1:2] * rgb[:, 2:]
                ], axis=1)

        elif model_name[-1] == '2':
            rgb_expanded = np.concatenate([
                    rgb,
                    rgb**2,
                    rgb[:, :1] * rgb[:, 1:2],
                    rgb[:, 1:2] * rgb[:, 2:],
                    rgb[:, :1] * rgb[:, 2:]
                ], axis=1)


    return rgb_expanded


def apply_matrix(
    rgb : npt.NDArray[np.float_],
    matrix: npt.NDArray[np.float_]   
) -> npt.NDArray[np.float_]:
    """
    Parameters
    ----------
    rgb : npt.NDArray[np.float_]
        Array of n x 3 shape that includes colors in RGB space of first camera.

    matrix : npt.NDArray[np.float_]
        Matrix that applies color correction.

    Returns
    -------
    rgb_out : npt.NDArray[np.float_]
        Array of n x 3 shape that includes colors in RGB space of second camera.

    """
    
    matrix_shape = matrix.shape

    if matrix_shape == (3, 9):
        rgb_expanded = get_poly_coords(rgb, "PCC2")
    
    elif matrix_shape == (3, 19):
        rgb_expanded = get_poly_coords(rgb, "PCC3")

    elif matrix_shape == (3, 6):
        rgb_expanded = get_poly_coords(rgb, "RPCC2")

    elif matrix_shape == (3, 13):
        rgb_expanded = get_poly_coords(rgb, "RPCC3")
    else:
        rgb_expanded = rgb

    rgb_out = rgb_expanded @ matrix.T

    return rgb_out