

def concatenate_decoded_csvs():
    files = "${sep=' ' decoded_csvs}".strip().split()

    import pandas as pd

    # get a non-zero size seed dataframe
    for i, f in enumerate(files):
        first = pd.read_csv(f, dtype={"target": object})
        if first.shape[0] != 0:
            break

    for f in files[i + 1:]:
        next_ = pd.read_csv(f, dtype={"target": object})

        # don't concatenate if the df is empty
        if next_.shape[0] != 0:
            first = pd.concat([first, next_], axis=0)

    # label spots sequentially
    first = first.reset_index.drop("index", axis=1)

    first.to_csv("decoded_concatenated.csv")