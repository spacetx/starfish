mkdir -p /tmp/starfish/max_projected
mkdir -p /tmp/starfish/transforms
mkdir -p /tmp/starfish/registered
mkdir -p /tmp/starfish/filtered
mkdir -p /tmp/starfish/results

URL=https://d2nhj9g34unfro.cloudfront.net/20181005/ISS-TEST/experiment.json

starfish validate experiment $URL

starfish filter \
    -i @${URL}'[fov_001][primary]' \
    -o /tmp/starfish/max_projected/primary_images.json \
    MaxProj \
    --dims c --dims z

starfish learn_transform \
    -i /tmp/starfish/max_projected/primary_images.json \
    -o /tmp/starfish/transforms/transforms.json \
    Translation \
    --reference-stack @$URL'[fov_001][dots]' \
    --upsampling 1000 \
    --axes r

starfish apply_transform \
    -i @$URL'[fov_001][primary]' \
    -o /tmp/starfish/registered/primary_images.json \
    --transformation-list /tmp/starfish/transforms/transforms.json \
    Warp

starfish filter \
    -i /tmp/starfish/registered/primary_images.json \
    -o /tmp/starfish/filtered/primary_images.json \
    WhiteTophat \
    --masking-radius 15

starfish filter \
    -i @$URL'[fov_001][nuclei]'  \
    -o /tmp/starfish/filtered/nuclei.json \
    WhiteTophat \
    --masking-radius 15

starfish filter \
    -i @$URL'[fov_001][dots]'  \
    -o /tmp/starfish/filtered/dots.json \
    WhiteTophat \
    --masking-radius 15

starfish detect_spots \
    --input /tmp/starfish/filtered/primary_images.json \
    --output /tmp/starfish/results/spots.nc \
    --blobs-stack /tmp/starfish/filtered/dots.json \
    BlobDetector \
    --min-sigma 4 \
    --max-sigma 6 \
    --num-sigma 20 \
    --threshold 0.01

starfish segment \
    --primary-images /tmp/starfish/filtered/primary_images.json \
    --nuclei /tmp/starfish/filtered/nuclei.json \
    -o /tmp/starfish/results/label_image.png \
    Watershed \
    --nuclei-threshold .16 \
    --input-threshold .22 \
    --min-distance 57

starfish target_assignment \
    --label-image /tmp/starfish/results/label_image.png \
    --intensities /tmp/starfish/results/spots.nc \
    --output /tmp/starfish/results/targeted-spots.nc \
    Label

starfish decode \
    -i /tmp/starfish/results/targeted-spots.nc \
    --codebook @$URL \
    -o /tmp/starfish/results/decoded-spots.nc \
    PerRoundMaxChannelDecoder

