
task process_field_of_view {

    String experiment
    Int field_of_view

    command <<<
        pip install git+https://github.com/spacetx/starfish.git@saxelrod-smFISH-wdl

        python3 <<CODE

        import starfish
        from starfish.core.pipelines import merfish_reciepe

        merfish_reciepe.process_fov(${field_of_view}, "${experiment}")


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
        from starfish.core.pipelines.wdl_utils import concatenate_decoded_csvs

        concatenate_decoded_csvs()
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


workflow ProcessMERFISH{

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
