task register {
  File input_tarball
  Int upsampling
  
  command <<<
    mkdir inputs
    tar xf '${input_tarball}' -C inputs --strip-components 1

    mkdir registered

    starfish register inputs/org.json registered --u '${upsampling}'
  >>>

  runtime {
    docker: "marcusczi/starfish:latest"
  }

  output {
    Array[File] registered_collection = glob("registered/*")
  }
}

task filter {
  
  Array[File] input_collection
  Int disk_size

  command <<<
    mkdir inputs
    for input_file in ${sep=' ' input_collection}; do
      mv $input_file inputs/
    done

    mkdir filtered

    starfish filter inputs/org.json filtered --ds '${disk_size}'
  >>>

  runtime {
    docker: "marcusczi/starfish:latest"
  }

  output {
    Array[File] filtered_collection = glob("filtered/*")
  }
}

task detect_spots {

  Array[File] input_collection
  Int min_sigma
  Int max_sigma
  Int num_sigma
  Float threshold

  command <<<
    mkdir inputs
    for input_file in ${sep=' ' input_collection}; do
      mv $input_file inputs/
    done

    mkdir detected

    starfish detect_spots inputs/org.json detected dots --min_sigma '${min_sigma}' \
      --max_sigma '${max_sigma}' --num_sigma '${num_sigma}' --t '${threshold}'
  >>>

  runtime {
    docker: "marcusczi/starfish:latest"
  }

  output {
    File spots_geo = "detected/spots_geo.csv"
    File encoder_table = "detected/encoder_table.csv"
  }
}

task segment {

  Array[File] input_collection
  File spots_geo
  File encoder_table
  Float dapi_threshold
  Float stain_threshold
  Int minimum_distance

  command <<<
    mkdir inputs
    for input_file in ${sep=' ' input_collection}; do
      mv $input_file inputs/
    done

    mkdir segmented
    mv '${spots_geo}' segmented
    mv '${encoder_table}' segmented


    starfish segment inputs/org.json segmented stain --dt '${dapi_threshold}' \
      --st '${stain_threshold}' --md '${minimum_distance}'
  >>>

  runtime {
    docker: "marcusczi/starfish:latest"
  }

  output {
    Array[File] segmented_collection = glob("segmented/*")
  }

}

task decode {
  Array[File] input_collection
  String decoder_type

  command <<<
    mkdir inputs
    for input_file in ${sep=' ' input_collection}; do
      mv $input_file inputs/
    done

    starfish decode inputs --decoder_type '${decoder_type}'
  >>>

  runtime {
    docker: "marcusczi/starfish:latest"
  }

  output {
    Array[File] output_collection = glob("inputs/*")
  }
}


workflow starfish {
  
  Int upsampling
  Int disk_size
  Int min_sigma
  Int max_sigma
  Int num_sigma
  Float threshold
  Float dapi_threshold
  Float stain_threshold
  Int minimum_distance
  String decoder_type

  File input_tarball
  
  call register {
    input: input_tarball=input_tarball, upsampling=upsampling
  }

  call filter {
    input: input_collection=register.registered_collection, disk_size=disk_size
  }

  call detect_spots {
    input: input_collection=filter.filtered_collection, min_sigma=min_sigma,
      max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold
  }

  call segment {
    input: input_collection=filter.filtered_collection, dapi_threshold=dapi_threshold,
      stain_threshold=stain_threshold, minimum_distance=minimum_distance,
      spots_geo=detect_spots.spots_geo, encoder_table=detect_spots.encoder_table
  }

  call decode {
    input: input_collection=segment.segmented_collection, decoder_type=decoder_type
  }

  output {
    File spots_geo = detect_spots.spots_geo
    File encoder_table = detect_spots.encoder_table
    Array[File] decoded_results = decode.output_collection
  }
}
