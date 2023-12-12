# config/settings.py
class GlulamConfig:
    """ Glulam factory settings, given in mm """
    MIN_HEIGHT = {'Region1': 22, 'Region2': 22}
    MAX_HEIGHT = {'Region1': 27, 'Region2': 27}
    MIN_LENGTH = {'Region1': 2000, 'Region2': 2000}
    MAX_LENGTH = {'Region1': 25000, 'Region2': 16000}
    DEFAULT_DEPTH = 115  # in mm
