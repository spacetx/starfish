# from json import JSONEncoder
#
#
# class StarfishJSONEncoder(JSONEncoder):
#
#     def default(self, o):
#         try:
#             # if the object has a custom to_json method
#             return o.to_json()
#         except:
#             try:
#                 # Use regular
#                 return super(StarfishJSONEncoder, self).default(o)
#             except:
#                 # If all else fails just log the class name
#                 return o.__class__.__name__