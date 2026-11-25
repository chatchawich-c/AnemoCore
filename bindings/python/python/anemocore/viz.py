import importlib
from typing import Any

# Lazy import helper
def _lazy_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        raise ImportError("matplotlib is required for visualization. Install the 'viz' extra.")

def plot_wind_field(wind_speeds: Any, title: str = "Wind Speed Field") -> None:
    """Plot a 2‑D wind speed field using matplotlib.
    The function imports matplotlib only when called, keeping the base package
    lightweight. ``wind_speeds`` should be a 2‑D array‑like object.
    """
    plt = _lazy_matplotlib()
    fig, ax = plt.subplots()
    ax.imshow(wind_speeds, cmap="viridis", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("X index")
    ax.set_ylabel("Y index")
    plt.colorbar(ax.images[0], ax=ax, label="Speed (m/s)")
    plt.show()

def plot_aep_series(aep_series: Any, title: str = "Annual Energy Production") -> None:
    """Plot AEP versus time or scenario index.
    ``aep_series`` can be any 1‑D iterable of numbers.
    """
    plt = _lazy_matplotlib()
    fig, ax = plt.subplots()
    ax.plot(aep_series, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("AEP (MWh)")
    plt.grid(True)
    plt.show()
