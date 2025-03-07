class DataPathMissingError(ValueError):
    """Data for model training missing error."""

    def __init__(self, path)->None:
        msg = f"Data path is required for model training"
        super().__init__(msg)

