
task process_field_of_view {

    String experiment
    Int field_of_view
    String recipe_file

    command <<<
        wget -O recipe.py ${recipe_file}

        python3 <<CODE

        import starfish
        recipe = __import__('recipe')

        decoded_spots = recipe.process_fov(${field_of_view}, "${experiment}")
        decoded_spots.save_csv("decoded.csv")

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


workflow Starfish {
    Int num_fovs
    String experiment
    String recipe_file

    Array[Int] fields_of_view = range(num_fovs)

    scatter(fov in fields_of_view) {
        call process_field_of_view {
            input:
                experiment = experiment,
                field_of_view = fov,
                recipe_file = recipe_file
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
