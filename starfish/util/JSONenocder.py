from json import JSONEncoder


class LogEncoder(JSONEncoder):
    def default(self, o):
        try:
            return super(LogEncoder, self).default(o)
        except TypeError:
            return JSONEncoder().encode(repr(o))
