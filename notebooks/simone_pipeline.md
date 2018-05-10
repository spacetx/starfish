# Summary of Simone's pipeline

## Overview of methods needed

1. 3d gaussian filter (`skimage.filters.gaussian` or `scipy.ndimage.gaussian`) -> `starfish.filters.gaussian` (already present)
2. `ndimage.gaussian_laplace` -> `starfish.filters.gaussian_laplace`
	1. Can we assume that any trimming of the image to fit within the dynamic range should be associated with the function that produces the aberration? E.g. `gaussian_laplace` can produce negative values -- should we wrap that function so that it doesn't have this characteristic?
  2. If no, we will need some clean-up functions
3. `max_projection` over z -> `starfish.image._stack` (currently present)
4. Threshold finding (is there an equivalent library function for the gradient calculation + trimming that this func does? -> `starfish.stats.calculate_peak_threshold_abs`. Replace `calculate` with your favorite word.
5. `peak_local_max` -> `starfish.spots.peakfinder` (location from brian-long)

## The pipeline:

- First, `process_standalone_experiment.py` is called. 
- This is primarily about munging files, and loads in `counting.filtering_and_counting` to do the work.

### Arguments:

1. path - path to experiment
2. analysis name - prefix for experiment
3. stringency (default 0) - ? TODO
4. min-distance (default 5) - minimum distance between peaks
5. min-plane (default None) - allows selection of a plane to start from (exclude out-of-focus)
6. max-plane (default None) - allows selection of a plane to end at (exclude out-of-focus)

- Next, `counting.filtering_and_counting` is called

### Arguments:

List Simone's, they're well documented. 

```
fpath_img_to_filter: str
path to the file to process
filtered_png_img_gene_dirs: list
list of the paths of the directories where the filtered images as are
saved as pngs.
filtered_img_gene_dirs: list
list of the paths of the directories where the filtered images are saved
as .npy.
counting_gene_dirs: list
list of the paths of the directories where the countings of the filtered
images are saved.
illumination_correction: bool
if True the illumination correction is run on the dataset.
plane_keep: list
start and end point of the z-planes to keep. Default None
keep all the planes (ex. [2,-3]).
min_distance: int
minimum distance between dots. (peaks)
stringency: int
stringency use to select the threshold used for counting. (default 0)
skip_genes_counting: list
list of the genes to skip for counting count.
skip_tags_counting: list
list of the tags inside the genes/stainings name to avoid to count.
```

Procedure: 

Note: Expects np.float64 images (how to address this? will everything work?)

1. `filtering.nuclei_filtering`
	1. called if this is in part of skip-genes, designed to skip nuclei, I think
  2. 3d gaussian filter, sigma (2, 100, 100) (z, y, x) (could be z, x, y), but x <-> y for our purposes.
  3. set negative values to zero
  4. max project over z (np.amax(stack, axis=0)
2. otherwise, call `filtering.smFISH_filtering`
  1. `gaussian_filter`, sigma=(1, 8, 8)
  2. `gaussian_laplace(stack, (0.2, 0.5, 0.5)`  # todo look up
    3. this makes the peaks negative, so invert signal `img_stack = -img_stack`
    4. ... and zero out the negative values again
    5. max project over z `(np.amax(img_stack, axis=0)`
3. `dots_calling.thr_calculator`
  1. Counts dots across the whole image using `img_filtered` results, `min_distance`, and `stringency`

### Thr Calculator

This method calculates a threshold to determine if dots are peaks or not. Specifically, it estimates
threshold_abs for `skimage.feature.peak_local_max`

Arguments:
1. `min_distance`: if two peaks are not > 5 pixels apart they are the same peak
2. stringency: higher stringency == higher threshold from `thr_array` to get peaks (tuning parameter)

Returns:

Lots of stuff in the form of a dictionary
1. `selected_thr`: threshold for counting after application of stringency
2. `calculated_thr`: threshold after all the trimming is calculated but before stringency application
3. `selected_peaks`: 2d coordinates of the peaks
4. `thr_array`: array of 100 points between Img.min() and Img.max(); this is a linspace between min and max
5. `peaks_coords`: coordinates of peaks at each threshold value.
6. `total peaks`: number of peaks at each threshold
7. `thr_idx`: index of the calculated threshold (some value within Thr index)

Procedure:
- for each of 100 thresholds between min and max, call peaks with `peak_local_max`
- when number of peaks is <= 3, stop
- if there's only one section with > 3 peaks, return No peaks, else:
- identify the threshold that was the stopping point
- calculate the gradient of the number of peaks detected at each threshold
- identify the minimum gradient value, and remove (1) all threshold values below that peak and all peaks before that value.
- if there's at least one peak left:
- create a line joining the ends of the gradient
- get the distances of all points from the line
- throw out the end points
- throw out points from both ends == stringency value (simone thinks this method oversamples)
- create a mask at the selected threshold (from before)
- do some magic where a mask is examined for large and small objects (area < 6 or area > 200) which are removed

```
# Threshold the image using the selected threshold
if selected_thr>0:
		img_mask = filtered_img>selected_thr

labels = nd.label(img_mask)[0]

properties = measure.regionprops(labels)

for ob in properties:
		if ob.area<6 or ob.area>200:
				img_mask[ob.coords[:,0],ob.coords[:,1]]=0

labels = nd.label(img_mask)[0]
selected_peaks = feature.peak_local_max(filtered_img, min_distance=min_distance, threshold_abs=selected_thr, exclude_border=False, indices=True, num_peaks=np.inf, footprint=None, labels=labels)
```

- then at the end, call `peak_local_max` on the `filtered_image` (original input to this function) with `threshold_abs` as selected threshold.
- get intensites by extracting the values that correspond to the peaks returned by peak local max
`selected_peaks_int = filtered_img[selected_peaks[:,0],selected_peaks[:,1]]`

- then, convert the float image to uint16 and save the filtered image
