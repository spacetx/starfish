
task process_field_of_view {

    String experiment
    Int field_of_view

    command <<<

        python3 <<CODE
        import sys

        import starfish

        # bandpass filter to remove cellular background and camera noise
        bandpass = starfish.image.Filter.Bandpass(lshort=.5, llong=7, threshold=0.0)

        # gaussian blur to smooth z-axis
        glp = starfish.image.Filter.GaussianLowPass(
            sigma=(1, 0, 0),
            is_volume=True
        )

        # pre-filter clip to remove low-intensity background signal
        clip1 = starfish.image.Filter.Clip(p_min=50, p_max=100)

        # post-filter clip to eliminate all but the highest-intensity peaks
        clip2 = starfish.image.Filter.Clip(p_min=99, p_max=100)

        # peak finding
        tlmpf = starfish.spots.DetectSpots.TrackpyLocalMaxPeakFinder(
            spot_diameter=5,
            min_mass=0.02,
            max_size=2,
            separation=7,
            noise_size=0.65,
            preprocess=False,
            percentile=10,
            verbose=True,
            is_volume=True,
        )

        def processing_pipeline(experiment, fov_name):

            print("Loading images...", file=sys.stderr)
            primary_image = experiment[fov_name].get_image(starfish.FieldOfView.PRIMARY_IMAGES)

            filter_kwargs = dict(in_place=True, verbose=True)

            all_intensities = list()
            for primary_image in experiment[fov_name].iterate_image_type(starfish.FieldOfView.PRIMARY_IMAGES):

                print("Applying Clip...", file=sys.stderr)
                clip1.run(primary_image, **filter_kwargs)

                print("Applying Bandpass...", file=sys.stderr)
                bandpass.run(primary_image, **filter_kwargs)

                print("Applying Gaussian low pass...", file=sys.stderr)
                glp.run(primary_image, **filter_kwargs)

                print("Applying Clip...", file=sys.stderr)
                clip2.run(primary_image, **filter_kwargs)

                print("Calling spots...", file=sys.stderr)
                spot_attributes = tlmpf.run(primary_image)

                all_intensities.append(spot_attributes)

            spot_attributes = starfish.IntensityTable.concatenate_intensity_tables(all_intensities)

            print("Decoding spots...", file=sys.stderr)
            decoded = experiment.codebook.decode_per_round_max(spot_attributes)
            decoded = decoded[decoded["total_intensity"] > .025]

            return primary_image, decoded

        # process cromwell inputs

        # translate field of view integer into a string key
        fov: int = ${field_of_view}
        fov_str: str = f"fov_{fov:03d}"

        # load experiment
        experiment = starfish.Experiment.from_json("${experiment}")

        _, decoded = processing_pipeline(experiment, fov_str)

        print("Writing decoded spots...", file=sys.stderr)
        df = decoded.to_decoded_spots()
        df.to_csv("decoded.csv")

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

        files = "${sep=', ' decoded_csvs}"
        import pandas as pd

        first = pd.read_csv(files[0], index_col=0)

        for f in files[1:]:
            next = pd.read_csv(f, index_col=0)
            first = pd.concat(first, f, axis=0)

        first.to_csv("decoded_concatenated.csv")

        CODE
    >>>

    runtime {
        docker: "spacetx/starfish:0.1.0-simple"
        memory: "16 GB"
        cpu: "2"
        disk: "local-disk 100 SDD"
    }

    output {
        File decoded_spots = "decoded_concatenated.csv"
    }
}


workflow ProcessSequentialSMFISH {

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
