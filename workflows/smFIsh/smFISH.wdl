
task process_field_of_view {

    String experiment
    Int field_of_view

    command <<<
        pip install git+https://github.com/spacetx/starfish.git@saxelrod-smFISH-wdl

        python3 <<CODE
        import starfish
        from starfish import IntensityTable
        from starfish.types import Axes
        from starfish.core.imagestack import indexing_utils

        fov: int = ${field_of_view}
        fov_str: str = f"fov_{int(fov):03d}"

        # load experiment
        print("loading experiment")
        experiment = starfish.Experiment.from_json("${experiment}")

        fov = experiment[fov_str]
        print("loainf fov")
        imgs = fov.get_images(starfish.FieldOfView.PRIMARY_IMAGES, zplanes=[1, 2, 3])
        all_decoded = list()
        for i, img in enumerate(imgs):
            print(f"processin fov {fov_str} round {i}...")
            codebook = indexing_utils.index_keep_dimensions(experiment.codebook, {Axes.ROUND: i})

            # filter
            print("clipping")
            clip1 = starfish.image.Filter.Clip(p_min=50, p_max=100)
            clip1.run(img)

            print("bandpass")
            bandpass = starfish.image.Filter.Bandpass(lshort=.5, llong=7, threshold=0.0)
            bandpass.run(img)

            print("gaussian")
            glp = starfish.image.Filter.GaussianLowPass(sigma=(1, 0, 0), is_volume=True)
            glp.run(img)


            print("clipping")
            clip2 = starfish.image.Filter.Clip(p_min=99, p_max=100, is_volume=True)
            clip2.run(img)

            print("detecting")
            tlmpf = starfish.spots.DetectSpots.TrackpyLocalMaxPeakFinder(
                spot_diameter=5,  # must be odd integer
                min_mass=0.02,
                max_size=2,  # this is max radius
                separation=7,
                noise_size=0.65,  # this is not used because preprocess is False
                preprocess=False,
                percentile=10,  # this is irrelevant when min_mass, spot_diameter, and max_size are set properly
                verbose=True,
                is_volume=True,
            )

            intensities = tlmpf.run(img)

            print("decoding")
            decoded = codebook.decode_per_round_max(intensities)
            all_decoded.append(decoded)

        print("concatenating decoded spots for {fov_str}")
        decoded = IntensityTable.concatenate_intensity_tables(all_decoded)
        # save results
        df = decoded.to_decoded_spots()
        print("saving csv")
        df.save_csv("decoded.csv")
        CODE
    >>>

    runtime {
        docker: "spacetx/starfish:0.1.0-simple"
        memory: "16 GB"
        cpu: "4"
        disk: "local-disk 100 SDD"
    }

    output {
        File decoded_csv = "decoded.csv"
    }
}


task concatenate_fovs {
    Array[File] decoded_csvs

    command <<<
        python <<CODE

        print("concatenating")
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

        CODE
    >>>

    runtime {
        docker: "spacetx/starfish:0.1.0-simple"
        memory: "16 GB"
        cpu: "1"
        disk: "local-disk 100 SDD"
    }

    output {
        File decoded_spots = "decoded_concatenated.csv"
    }
}


workflow ProcessSmFISH{

    Int num_fovs
    String experiment

    Array[Int] fields_of_view = range(num_fovs)

    scatter(fov in fields_of_view) {
        call process_field_of_view {
            input:
                experiment = experiment,
                field_of_view = fov
        }
    }

    call concatenate_fovs {
        input:
            decoded_csvs = process_field_of_view.decoded_csv
    }

    output {
        File decoded_spots = concatenate_fovs.decoded_spots
    }
}
