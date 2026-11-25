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

# New lazy‑load visualization helpers
def bearing_health_trend(data: Any, **kwargs) -> Any:
    """Plot bearing health trend.
    ``data`` should be a sequence of health metric values.
    ``matplotlib`` is imported lazily.
    """
    plt = _lazy_matplotlib()
    fig, ax = plt.subplots()
    ax.plot(data, **kwargs)
    ax.set_title("Bearing Health Trend")
    ax.set_xlabel("Time Index")
    ax.set_ylabel("Health Metric")
    plt.grid(True)
    plt.show()
    return fig

def economic_risk_fan_chart(risk_data: dict, **kwargs) -> Any:
    """Plot economic risk fan chart.
    ``risk_data`` is a mapping with keys like ``'percentiles'`` (list of values)
    and ``'scenario'`` (list of scenario identifiers).
    """
    plt = _lazy_matplotlib()
    fig, ax = plt.subplots()
    percentiles = risk_data.get('percentiles', [])
    scenario = risk_data.get('scenario', range(len(percentiles)))
    ax.fill_between(scenario, percentiles, step='pre', alpha=0.3, **kwargs)
    ax.set_title("Economic Risk Fan Chart")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Risk Metric")
    plt.grid(True)
    plt.show()
    return fig
