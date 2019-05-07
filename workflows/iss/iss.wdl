
task process_field_of_view {

    String experiment
    Int field_of_view

    command <<<
        python3 <<CODE

        import numpy as np
        import starfish
        from starfish.types import Axes

        fov: int = ${field_of_view}
        fov_str: str = f"fov_{int(fov):03d}"

        # load experiment
        experiment = starfish.Experiment.from_json("${experiment}")

        fov = experiment[fov_str]
        imgs = fov.get_image(starfish.FieldOfView.PRIMARY_IMAGES)
        dots = imgs.max_proj(Axes.CH)

        # filter
        filt = starfish.image.Filter.WhiteTophat(masking_radius=15, is_volume=False)
        filtered_imgs = filt.run(imgs, verbose=True, in_place=False)
        filt.run(dots, verbose=True, in_place=True)

        # find threshold
        tmp = dots.sel({Axes.ROUND:0, Axes.CH:0, Axes.ZPLANE:0})
        dots_threshold = np.percentile(np.ravel(tmp.xarray.values), 50)

        # find spots
        p = starfish.spots.DetectSpots.BlobDetector(
            min_sigma=1,
            max_sigma=10,
            num_sigma=30,
            threshold=dots_threshold,
            measurement_type='mean',
        )

        # blobs = dots; define the spots in the dots image, but then find them again in the stack.
        intensities = p.run(filtered_imgs, blobs_image=dots, blobs_axes=(Axes.ROUND, Axes.ZPLANE))

        # decode
        decoded = experiment.codebook.decode_per_round_max(intensities)

        # save results
        df = decoded.to_decoded_spots()
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


workflow ProcessISS{

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
