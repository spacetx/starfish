.. _logging:

Provenance Logging
===================

Every method run on an ImageStack is recorded on the stacks log attribute. Each entry includes

- Method name
- Arguments supplied to the method
- os information
- starfish version
- release tag


Example
--------
This is an example of the formatted provenance log after processing an ImageStack through the ISS pipeline.

.. code-block:: python

    >>>pprint(stack.log)

    [{'arguments': {'clip_method': '"<Clip.CLIP: \'clip\'>"',
                    'is_volume': False,
                    'masking_radius': 15},
      'dependencies': {'numpy': '1.16.1',
                       'pandas': '0.24.1',
                       'scikit-image': '0.14.2',
                       'scikit-learn': '0.20.2',
                       'scipy': '1.2.1',
                       'sympy': '1.3',
                       'xarray': '0.11.3'},
      'method': 'WhiteTophat',
      'os': {'Platform': 'Darwin',
             'Python Version': '3.6.5',
             'Version:': 'Darwin Kernel Version 17.7.0: Thu Jun 21 22:53:14 PDT '
                         '2018; root:xnu-4570.71.2~1/RELEASE_X86_64'},
      'release tag': 'Running starfish from source',
      'starfish version': '0.0.36+4.g169fe89b.dirty'},
     {'arguments': {},
      'dependencies': {'numpy': '1.16.1',
                       'pandas': '0.24.1',
                       'scikit-image': '0.14.2',
                       'scikit-learn': '0.20.2',
                       'scipy': '1.2.1',
                       'sympy': '1.3',
                       'xarray': '0.11.3'},
      'method': 'Warp',
      'os': {'Platform': 'Darwin',
             'Python Version': '3.6.5',
             'Version:': 'Darwin Kernel Version 17.7.0: Thu Jun 21 22:53:14 PDT '
                         '2018; root:xnu-4570.71.2~1/RELEASE_X86_64'},
      'release tag': 'Running starfish from source',
      'starfish version': '0.0.36+4.g169fe89b.dirty'},
     {'arguments': {},
      'dependencies': {'numpy': '1.16.1',
                       'pandas': '0.24.1',
                       'scikit-image': '0.14.2',
                       'scikit-learn': '0.20.2',
                       'scipy': '1.2.1',
                       'sympy': '1.3',
                       'xarray': '0.11.3'},
      'method': 'Warp',
      'os': {'Platform': 'Darwin',
             'Python Version': '3.6.5',
             'Version:': 'Darwin Kernel Version 17.7.0: Thu Jun 21 22:53:14 PDT '
                         '2018; root:xnu-4570.71.2~1/RELEASE_X86_64'},
      'release tag': 'Running starfish from source',
      'starfish version': '0.0.36+4.g169fe89b.dirty'},
     {'arguments': {'detector_method': '"<function blob_log at 0x1233208c8>"',
                    'is_volume': True,
                    'max_sigma': 10,
                    'measurement_function': '"<function mean at 0x10c41c378>"',
                    'min_sigma': 1,
                    'num_sigma': 30,
                    'overlap': 0.5,
                    'threshold': 0.01},
      'dependencies': {'numpy': '1.16.1',
                       'pandas': '0.24.1',
                       'scikit-image': '0.14.2',
                       'scikit-learn': '0.20.2',
                       'scipy': '1.2.1',
                       'sympy': '1.3',
                       'xarray': '0.11.3'},
      'method': 'BlobDetector',
      'os': {'Platform': 'Darwin',
             'Python Version': '3.6.5',
             'Version:': 'Darwin Kernel Version 17.7.0: Thu Jun 21 22:53:14 PDT '
                         '2018; root:xnu-4570.71.2~1/RELEASE_X86_64'},
      'release tag': 'Running starfish from source',
      'starfish version': '0.0.36+4.g169fe89b.dirty'}]

