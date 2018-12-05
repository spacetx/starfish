from json import JSONEncoder


class StarfishJSONEncoder(JSONEncoder):

    def default(self, o):
        try:
            return o.to_json()
        except:
            try:
                return super(StarfishJSONEncoder, self).default(o)
            except:
                return o.__class__.__name__