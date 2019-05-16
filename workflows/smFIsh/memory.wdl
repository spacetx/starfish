#version 1.0

task check_mem {

    command <<<
        cat /proc/meminfo
    >>>

    runtime {
        docker: "spacetx/starfish:0.1.0-simple"
        memory: "200 GB"
        cpu: "2"
        disk: "local-disk 100 SDD"
    }

}


workflow ProcessSmFISH{
    Int num_fovs
    Int num_rounds
    String experiment

    Array[Int] fields_of_view = range(num_fovs)
    Array[Int] rounds_per_fov = range(num_rounds)
    # maybe try this after if things are weird
    Array[Pair[Int, Int]] crossed = cross(fields_of_view, rounds_per_fov)

    scatter(fov_round in crossed) {
        Int fov = fov_round.left
        Int round = fov_round.right
        call check_mem {
        }

    }

}
