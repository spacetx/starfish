import numpy as np
import pandas as pd

from starfish.munge import dataframe_to_multiindex


def test_dataframe_from_multiindex():
    data = np.arange(4).reshape(2, 2)
    names = list('cd')
    df = pd.DataFrame(data, index=list('ab'), columns=names)
    mi = dataframe_to_multiindex(df)

    assert mi.names == names
    assert len(mi.levels) == 2
    assert np.array_equal(mi.get_level_values('c'), np.array([0, 2]))
    assert np.array_equal(mi.get_level_values('d'), np.array([1, 3]))
