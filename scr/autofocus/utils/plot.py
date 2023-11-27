from typing import Union


def set_size(
    text_width: Union[float, str] = "paper",
    fraction: float = 1.0,
    subplots=(1, 1),
    scale_factor: float = 1.0,
    rescale_height: float = 1.0,
):
    """Set figure dimensions to avoid scaling in LaTeX.

    Based largely on Jack Walton's post on ploting figures with matplotlib and LaTeX:
    https://jwalton.info/Embed-Publication-Matplotlib-Latex/

    Parameters
    ----------
        text_width: float or string
                Document width in points, or string of predefined document type.
        fraction: float, optional
                Fraction of the width which you wish the figure to occupy.
        subplots: array-like, optional
                The number of rows and columns of subplots.
        scale_factor: float
            Facto to scale width and height with.
        rescale_height: float
            Factor to rescale height.

    Returns
    -------
        fig_dim: tuple
                Dimensions of figure in inches
    """
    if text_width == "paper":
        # Textwidth of LaTeX file. Can be determined by typing
        # \the\text_width
        # in your latex file and then compiling.
        width_pt = 483.69687
    elif text_width == "beamer":
        width_pt = 307.28987
    elif text_width == "presentation":
        width_pt = 600
    elif isinstance(text_width, (float, int)):
        width_pt = text_width
    else:
        raise ValueError("Textwidth has to be 'paper', 'beamer', 'presentation' or a float.")

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (scale_factor * fig_width_in, rescale_height * scale_factor * fig_height_in)
