class Singleton(type):
    """Metaclass to make TestComponent a singleton. Not strictly necessary."""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def clean_namespace(retain):
    for name in globals():
        if name not in retain:
            del globals()[name]

