# napari-foobar

[![License MIT](https://img.shields.io/pypi/l/napari-foobar.svg?color=green)](https://github.com/githubuser/napari-foobar/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-foobar.svg?color=green)](https://pypi.org/project/napari-foobar)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-foobar.svg?color=green)](https://python.org)
[![tests](https://github.com/githubuser/napari-foobar/workflows/tests/badge.svg)](https://github.com/githubuser/napari-foobar/actions)
[![codecov](https://codecov.io/gh/githubuser/napari-foobar/branch/main/graph/badge.svg)](https://codecov.io/gh/githubuser/napari-foobar)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-foobar)](https://napari-hub.org/plugins/napari-foobar)

A simple plugin to use Cellpose segmentation within napari

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.


https://github.com/MatthewMing11/napari-foobar/assets/50256336/8c054b57-ff9c-44ad-85d9-e85859c6cc42


<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

Before installing the plugin, make sure to create an environment, either in conda or mamba.
If you don't have mamba installed, replace mamba by conda.
    
    mamba create -n napari-plugin-env python=3.9 napari jupyterlab -c conda-forge
    mamba activate napari-plugin-env

To install mamba, go [here](https://mamba.readthedocs.io/en/latest/installation.html) and follow the instructions under "Fresh install".

Mamba has been shown to run significantly faster than running with conda, so consider using Mamba.

Then do

    git clone https://github.com/MatthewMing11/napari-foobar.git

Make sure you know where you downloaded the project as you will need to install it locally.

You can install `napari-foobar` via [pip] locally. Navigate inside the napari-foobar folder and paste the command:

    pip install -e .

When making updates/receiving updates, make sure to 

    pip install -e .

so that all files are updated on the plugin.

## Components
![](https://github.com/MatthewMing11/napari-foobar/blob/main/record.gif)
### image_layer

Requires a valid image layer to use cellpose features. Not required if not doing segmentation.

### label_layer

Requires a valid label layer to use relabeling features. Not required if not doing relabeling.

### diameter

The length that will be used to approximate and segment cells.

### anisotropy

The amount of deformity on z compared to x and y on voxels used.

### min size

The lower limit of cells allowed to be segmented. Anything lower is excluded from segmentation.

### pixelX, pixelY, pixelZ

The dimensions of the voxels

### resize cells from image

Resizes the cells so that they appear properly in the display.(Will only modify image selected in image_layer so if you have a labeled layer, change it to an image layer,select it under image_layer and process it with this button. Then convert it back to a label.)

### process stack as 3D

Enable if using 3D stack otherwise disable.

### clear previous results

Enable if newer cellpose results should replace old results.

### merge cells from image

(Optional) Merge undersegmented cells from image. (only works on label selected on label_layer)

### delete edge cells from image

Click to remove all cells on the border.(only works on label selected on label_layer)

### isolate cells from image

When starting to relabel, click this to setup as well as isolate the largest cell. (only works on label selected on label_layer). Make sure to press the square icon located on the lower left corner of napari before isolating so that isolation is done in 3D as it is 2D by default. 

### erode cells from image

Erodes selected cell. (only works on label selected on label_layer)

### watershed cells from image.

Watersheds selected cell. (only works on label selected on label_layer)

### delete cells from image

Deletes selected cell and moves to next cell on the list. (only works on label selected on label_layer)

### prev cell to relabel

Move to the previous largest labeled cell. (only works on label selected on label_layer)

### next cell to relabel

Move to the next largest labeled cell. (only works on label selected on label_layer)

### run segmentation

Runs cellpose on image given on image_layer.

## Potential issue with PyTorch

Cellpose and therefore the plugin and napari can crash without warning in some cases with torch==1.12.0. This can be fixed by reverting to an earlier version using:

    pip install torch==1.11.0

## Known Issues

The package numba sometimes doesn't work with the current numpy version. If so, please use

    pip install numpy==1.21.4

### typeerror
Because of a recent change to update everything in napari, there will be a "typeerror: issubclass() arg 1 must be a class" error. To resolve this, please use

    pip install typing-extensions==4.5.0

### peak_local_max error
peak_local_max will cause an error if you do not have the correct data type for the layer being relabeled for erosion and watershed. To change the data type, please right-click and select convert date type and change it to int64.

### index error when using erode,watershed,delete
This error is caused because the colors are not updated when new labels are made. This is resolved by clicking the show selected checkbox on and off to recalibrate the colors so that the new labels have colors.

## Resources/Links

[napari](https://napari.org/stable/)

[Code for binary fill algorithm](https://github.com/imagej/ImageJ/blob/master/ij/plugin/filter/Binary.java)

[cellpose plugin base](https://github.com/MouseLand/cellpose-napari)

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-foobar" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
