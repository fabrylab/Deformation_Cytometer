# extended dictionary for dot access

# yes as usual give us the dot ditcs ...
class dotdict(dict):
    """
    enables .access on dicts
    """
    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr, None)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
