
task process_field_of_view {

    String experiment
    Int field_of_view

    command <<<
        python3 <<CODE

        import starfish
        from starfish import Experiment, FieldOfView
        from starfish.image import Filter
        from starfish.image import ApplyTransform, LearnTransform
        from starfish.spots import DetectSpots
        from starfish.types import Axes

        fov: int = ${field_of_view}
        fov_str: str = f"fov_{int(fov):03d}"

        # load experiment
        experiment = starfish.Experiment.from_json("${experiment}")

        print(f"loading fov: {fov_str}")
        fov = experiment[fov_str]

        # note the structure of the 5D tensor containing the raw imaging data
        imgs = fov.get_image(FieldOfView.PRIMARY_IMAGES)
        dots = fov.get_image("dots")
        nuclei = fov.get_image("nuclei")

        masking_radius = 15
        print("Filter WhiteTophat")
        filt = Filter.WhiteTophat(masking_radius, is_volume=False)

        filtered_imgs = filt.run(imgs, verbose=True, in_place=False)
        filt.run(dots, verbose=True, in_place=True)
        filt.run(nuclei, verbose=True, in_place=True)

        print("Learning Transform")
        learn_translation = LearnTransform.Translation(reference_stack=dots, axes=Axes.ROUND, upsampling=1000)
        transforms_list = learn_translation.run(imgs.max_proj(Axes.CH, Axes.ZPLANE))

        print("Applying transform")
        warp = ApplyTransform.Warp()
        registered_imgs = warp.run(filtered_imgs, transforms_list=transforms_list, in_place=False, verbose=True)

        print("Detecting")
        p = DetectSpots.BlobDetector(
            min_sigma=1,
            max_sigma=10,
            num_sigma=30,
            threshold=0.01,
            measurement_type='mean',
        )

        intensities = p.run(registered_imgs, blobs_image=dots, blobs_axes=(Axes.ROUND, Axes.ZPLANE))

        decoded = experiment.codebook.decode_per_round_max(intensities)
        df = decoded.to_decoded_spots()
        print("saving decoded.csv")
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
        first = first.reset_index().drop("index", axis=1)

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
