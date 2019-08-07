



# def concatenate_spot_attributes_to_intensities(
#         spot_attributes: SpotAttributes
# ) -> IntensityTable:
#     """
#
#     Parameters
#     ----------
#     spot_attributes : Sequence[Tuple[SpotAttributes, Dict[Axes, int]]]
#         A sequence of SpotAttribute objects and the indices (channel, round) that each object is
#         associated with.
#
#     Returns
#     -------
#     IntensityTable :
#         concatenated input SpotAttributes, converted to an IntensityTable object
#
#     """
#     ch_values: Sequence[int] = sorted(set(inds[Axes.CH] for _, inds in spot_attributes))
#     round_values: Sequence[int] = sorted(set(inds[Axes.ROUND] for _, inds in spot_attributes))
#
#     # this drop call ensures only x, y, z, radius, and quality, are passed to the IntensityTable
#     features_coordinates = all_spots.drop(['spot_id', 'intensity'], axis=1)
#
#     intensity_table = IntensityTable.zeros(
#         SpotAttributes(features_coordinates), ch_values, round_values,
#     )
#
#     i = 0
#     for attrs, inds in spot_attributes:
#         for _, row in attrs.data.iterrows():
#             selector = dict(features=i, c=inds[Axes.CH], r=inds[Axes.ROUND])
#             intensity_table.loc[selector] = row['intensity']
#             i += 1
#
#     return intensity_table