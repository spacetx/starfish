.. _processing_at_scale:

Processing With AWS
===================

This tutorial will walk you through how to create two AWS Batch jobs that will use your existing starfish pipeline and
apply it in parallel to your entire experiment. Before we begin make sure you've completed the following prerequisites:

Prerequisites
-------------
- Download the template files and script needed to set up and run your aws job :download:`here </_static/starfish-aws-templates.zip>`
- Create an aws account with access to the console `Create Account <https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/>`__
- Create a starfish pipeline for processing a singe field of view from :ref:`ImageStack` to :ref:`DecodedIntensityTable`
- Convert your dataset to SpaceTx format. :ref:`section_formatting_data`
- Make sure you have the awscli installed ``pip install awscli``

For this tutorial we will be working with a 15 field of view `ISS dataset <https://s3.amazonaws.com/spacetx.starfish.data.public/browse/formatted/iss/20190506/experiment.json>`_

Alright let's begin!

Set up your data
-----------------

Upload your Data
+++++++++++++++++

If your dataset is already uploaded to an s3 bucket you can skip this section.

Open the aws console and navigate to `s3 <https://console.aws.amazon.com/s3/home>`_.  Select `Create Bucket` and enter the name of your bucket.
We name ours `aws-processing-example`. Once your bucket is created navigate into it and create a new folder for your spacetx-dataset. We call ours
`iss-spacetx-formatted`:

.. figure:: /_static/images/aws-tutorial-figure1.png
   :align: center

We also create a folder to hold the results from our processing called `iss-spacetx-formatted-results`.

Now that our buckets are created it's time to sync our dataset to it. Open up terminal and navigate to the folder with your spacetx formatted dataset. Run the following command:

``aws s3 sync . s3://<PATH_TO_DATASET_FOLDER>``

Ours looks like this:

``aws s3 sync . s3://aws-processing-example/iss-spacetx-formatted/``

To test that our experiment has been properly uploaded we try loading it up with starfish:


>>> from starfish import Experiment
>>> e = Experiment.from_json("https://s3.amazonaws.com/aws-processing-example/iss-spacetx-formatted/experiment.json")
>>> e

::

    {fov_000: <starfish.FieldOfView>
      Primary Image: <slicedimage.TileSet (z: 1, c: 4, r: 4, x: 1390, y: 1044)>
      Auxiliary Images:
        nuclei: <slicedimage.TileSet (z: 1, c: 1, r: 1, x: 1390, y: 1044)>
        dots: <slicedimage.TileSet (z: 1, c: 1, r: 1, x: 1390, y: 1044)>
    fov_001: <starfish.FieldOfView>
      Primary Image: <slicedimage.TileSet (z: 1, c: 4, r: 4, x: 1390, y: 1044)>
      Auxiliary Images:
        nuclei: <slicedimage.TileSet (z: 1, c: 1, r: 1, x: 1390, y: 1044)>
        dots: <slicedimage.TileSet (z: 1, c: 1, r: 1, x: 1390, y: 1044)>
    fov_002: <starfish.FieldOfView>
      Primary Image: <slicedimage.TileSet (z: 1, c: 4, r: 4, x: 1390, y: 1044)>
      Auxiliary Images:
        nuclei: <slicedimage.TileSet (z: 1, c: 1, r: 1, x: 1390, y: 1044)>
        dots: <slicedimage.TileSet (z: 1, c: 1, r: 1, x: 1390, y: 1044)>
    fov_003: <starfish.FieldOfView>
      Primary Image: <slicedimage.TileSet (z: 1, c: 4, r: 4, x: 1390, y: 1044)>
      Auxiliary Images:
        nuclei: <slicedimage.TileSet (z: 1, c: 1, r: 1, x: 1390, y: 1044)>
        dots: <slicedimage.TileSet (z: 1, c: 1, r: 1, x: 1390, y: 1044)>
      ...,
    }


Looks good! Let's move on.


Create a Recipe File
+++++++++++++++++++++

From the template files downloaded open the file named `recipe.py`. The file should include the method signature ``def process_fov(fov: FieldOfView, codebook: Codebook) -> DecodedIntensityTable:``.
Within this method add your starfish pipeline code for processing a single field of view. The return value should be a :ref:`DecodedIntensityTable`. Here's what our recipe.py file looks like.

.. code-block:: python

    from starfish import Codebook, DecodedIntensityTable, FieldOfView
    from starfish.image import ApplyTransform, Filter, LearnTransform
    from starfish.spots import DecodeSpots, FindSpots
    from starfish.types import Axes, FunctionSource


    def process_fov(fov: FieldOfView, codebook: Codebook) -> DecodedIntensityTable:
        """Process a single field of view of ISS data
        Parameters
        ----------
        fov : FieldOfView
            the field of view to process
        codebook : Codebook
            the Codebook to use for decoding

        Returns
        -------
        DecodedSpots :
            tabular object containing the locations of detected spots.
        """

        # note the structure of the 5D tensor containing the raw imaging data
        imgs = fov.get_image(FieldOfView.PRIMARY_IMAGES)
        dots = fov.get_image("dots")
        nuclei = fov.get_image("nuclei")

        print("Learning Transform")
        learn_translation = LearnTransform.Translation(reference_stack=dots, axes=Axes.ROUND, upsampling=1000)
        transforms_list = learn_translation.run(imgs.reduce({Axes.CH, Axes.ZPLANE}, func="max"))

        print("Applying transform")
        warp = ApplyTransform.Warp()
        registered_imgs = warp.run(imgs, transforms_list=transforms_list, verbose=True)

        print("Filter WhiteTophat")
        filt = Filter.WhiteTophat(masking_radius=15, is_volume=False)

        filtered_imgs = filt.run(registered_imgs, verbose=True)
        filt.run(dots, verbose=True, in_place=True)
        filt.run(nuclei, verbose=True, in_place=True)

        print("Detecting")
        detector = FindSpots.BlobDetector(
            min_sigma=1,
            max_sigma=10,
            num_sigma=30,
            threshold=0.01,
            measurement_type='mean',
        )
        dots_max = dots.reduce((Axes.ROUND, Axes.ZPLANE), func="max", module=FunctionSource.np)
        spots = detector.run(image_stack=filtered_imgs, reference_image=dots_max)

        print("Decoding")
        decoder = DecodeSpots.PerRoundMaxChannel(codebook=codebook)
        decoded = decoder.run(spots=spots)
        return decoded

Upload your recipe to s3. To make things easy we upload our recipe file to the same directory our experiment dataset lives in.

``aws s3 cp recipe.py s3://aws-processing-example/iss-spacetx-formatted/``

Set up your Batch Jobs
----------------------

So now we have our data and recipe uploaded and ready to go in s3, let's move on to actually creating our processing jobs.
Our final workflow will be composed of two jobs:

- Process each Field of View in parallel using an AWS Batch Array Job
- Combine the results from each Field of View into one large DecodedIntensityTable using an AWS Batch Job


Create a custom IAM Role
+++++++++++++++++++++++++

Before we can register our jobs we need to set up an IAM role that has access to AWSBatchServices and
our newly created s3 bucket. Navigate to the `IAM console <https://console.aws.amazon.com/iam/home>`_ an select *Roles* from the left panel.
Click *Create Role* we've called ours `spacetx-batch-uploader`. From the list of available services to prevision your role with select *batch*. Then click through the rest of the wizard
using the default settings and create the role.

We also need to give this role read and write access to our newly created s3 bucket. To do this we make a new policy and attach it to the `spacetx-batch-uploader` role.

Select *policies* from the left hand panel and click *create policy*. Click on the JSON editor and paste in the following code:


::

    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ListObjectsInBucket",
                "Effect": "Allow",
                "Action": [
                    "s3:ListBucket"
                ],
                "Resource": [
                    "arn:aws:s3:::<YOUR BUCKET>"
                ]
            },
            {
                "Sid": "AllObjectActions",
                "Effect": "Allow",
                "Action": "s3:*Object",
                "Resource": [
                    "arn:aws:s3:::<YOUR BUCKET>/*"
                ]
            }
        ]
    }


Here's what ours looks like:

::

    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "ListObjectsInBucket",
                "Effect": "Allow",
                "Action": [
                    "s3:ListBucket"
                ],
                "Resource": [
                    "arn:aws:s3:::aws-processing-example"
                ]
            },
            {
                "Sid": "AllObjectActions",
                "Effect": "Allow",
                "Action": "s3:*Object",
                "Resource": [
                    "arn:aws:s3:::aws-processing-example/*"
                ]
            }
        ]
    }


Name your policy and save it, we named our `spacetx-batch-uploader`. Now navigate back to your new role and attach your s3 uploader policy.
Our `spacetx-batch` role summery now looks like this:

.. figure:: /_static/images/aws-tutorial-figure2.png
   :align: center

Note the ARN of your new role (circled in the image). You'll need it in the next few steps.


Register your Jobs
+++++++++++++++++++

Follow the `Getting Started Guide <http://docs.aws.amazon.com/batch/latest/userguide/Batch_GetStarted.html>`_ and ensure you have a valid job queue and compute environment. For this tutorial
we used the default parameters (our job queue is still called first-run-job-queue).

Here's what out Batch Dashboard looks like:

.. figure:: /_static/images/aws-tutorial-figure5.png
   :align: center

Alright now it's time to register our batch jobs. From the template files open up the file named `register-process-fov-job.json`. This file describes a batch job that will create an ec2 instance using the docker container `spacetx/process-fov`
that processes a specified single field of view using your recipe.py file. Replace the string "ADD ARN" with the aws ARN of the role you just created in the last step. Our file looks like this:

::

    {
      "jobDefinitionName": "process-fov",
      "type": "container",
      "containerProperties": {
        "jobRoleArn": "arn:aws:iam::422553907334:role/spacetx-batch",
        "image": "spacetx/process-fov",
        "vcpus": 1,
        "memory": 2500
      }
    }

NOTE: if your starfish processing is memory expensive you can adjust the allocated memory for each created instance using the `memory` parameter.

Then from the directory where this file lives run the following command:

``aws batch submit-job --cli-input-json file://register-process-fov-job.json``

You can check that your jobs had been successfully registered by navigating to the `Job Definitions page <https://console.aws.amazon.com/batch/home>`_.

Here's what our's looks like:

.. figure:: /_static/images/aws-tutorial-figure3.png
   :align: center

Now open the file named `register-merge-job.json`. This file describes a batch job that will create an ec2 instance using the docker container `spacetx/merge-batch-job` that merges together all your processed results into
one `DecodedIntensityTable`. Again replace the string "ADD ARN" with the aws ARN of your batch processing role. Our file looks like this:

::

    {
      "jobDefinitionName": "merge-job",
      "type": "container",
      "containerProperties": {
        "jobRoleArn": "arn:aws:iam::422553907334:role/spacetx-batch",
        "image": "spacetx/merge-batch-job",
        "vcpus": 1,
        "memory": 2500
      }
    }

Then from the directory where this file lives run the following command:

``aws batch submit-job --cli-input-json file://register-merge-job.json``

Again, check that your job has been successfully registered from the job console, our two jobs are ready to go!

.. figure:: /_static/images/aws-tutorial-figure4.png
   :align: center


Run your Batch Jobs
-------------------

Now that we've set everything up it's time to run our jobs! The script `starfish-workflow.py` will handle submitting the process-fov array job
then the merge job with a dependency on the first job to finish. All you'll need to do is run the script with a few parameters:

::

    --experiment-url: The path to your experiment.json file. Our is "s3://aws-processing-example/iss-spacetx-formatted/experiment.json"

    --num-fovs: The number of fields of view in the experiment. We have 15

    --recipe-location: The path to your recipe file in s3. Ours is "s3://aws-processing-example/aws-processing-example/iss-spacetx-formatted/recipe.py"

    --results-location: The s3 bucket to copy the results from the job to. Ours is "s3://aws-processing-example/iss-spacetx-formatted-results/"

    --job-queue: The name of your job queue to run your jobs. Ours is "first-run-job-queue"


Now we run our script:

::

    $ python3 starfish-workflow.py \
    >     --experiment-url "s3://aws-processing-example/iss-spacetx-formatted/experiment.json" \
    >     --num-fovs 15 \
    >     --recipe-location "s3://aws-processing-example/aws-processing-example/iss-spacetx-formatted/recipe.py" \
    >     --results-bucket "s3://aws-processing-example/iss-spacetx-formatted-results/" \
    >     --job-queue "first-run-job-queue"
    Process fovs array job 39a13edd-8cca-4e7e-9379-aa3cf757c72e successfully submitted.
    Merge results job ac5d49f5-a12e-4176-96e2-f697c6cf0a12 successfully submitted.

To monitor the status of both jobs navigate to the `AWS Batch Dashboard <https://console.aws.amazon.com/batch/home>`_. You should see 2
jobs under PENDING

.. figure:: /_static/images/aws-tutorial-figure6.png
   :align: center

From here you should be able to click on the jobs and track their movement through the RUNNABLE -> RUNNING -> SUCCEEDED states.
NOTE: Batch jobs may take up to 10 minutes to move from PENDING to RUNNABLE. When both jobs have reached the SUCCEEDED state check
that everything worked by navigating to your results bucket. The bucket should include the processed results from
each field of view as well as the concatenated results called `merged_decoded_fovs.nc`. Here's what our bucket contains:

.. figure:: /_static/images/aws-tutorial-figure7.png
   :align: center

And that's it! You have successfully set up and processed your experiment using aws. As long as you keep your job definitions you can rerun the jobs
using the same command anytime.
