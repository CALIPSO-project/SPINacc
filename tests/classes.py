class pack:
    """
    Variables that are packaged
    """

    def __init__(self):
        self.__doc__ = "packaged data"

    def __getitem__(self, key):
        return self.__dict__[key]


class auxiliary:
    """
    Class to store auxiliary variables
    """

    def __getitem__(self, key):
        return self.__dict__[key]

    def __init__(self):
        self.__doc__ = "auxiliary data"
