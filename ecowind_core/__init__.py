import anemocore as _ac

def __getattr__(name: str):
    """Delegate attribute access to the anemocore package.
    This provides compatibility for existing tests that import `ecowind_core`.
    """
    return getattr(_ac, name)
