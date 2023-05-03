# napari-foobar

[![License MIT](https://img.shields.io/pypi/l/napari-foobar.svg?color=green)](https://github.com/githubuser/napari-foobar/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-foobar.svg?color=green)](https://pypi.org/project/napari-foobar)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-foobar.svg?color=green)](https://python.org)
[![tests](https://github.com/githubuser/napari-foobar/workflows/tests/badge.svg)](https://github.com/githubuser/napari-foobar/actions)
[![codecov](https://codecov.io/gh/githubuser/napari-foobar/branch/main/graph/badge.svg)](https://codecov.io/gh/githubuser/napari-foobar)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-foobar)](https://napari-hub.org/plugins/napari-foobar)

A simple plugin to use FooBar segmentation within napari

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-foobar` via [pip]:

    pip install napari-foobar

To install latest development version :

    pip install git+https://github.com/MatthewMing11/napari-foobar

Before installing the plugin, make sure to create an environment, either in conda or mamba.
If you don't have mamba installed, replace mamba by conda.
    
    mamba create -n napari-plugin-env python=3.9 napari jupyterlab -c conda-forge
    mamba activate napari-plugin-env

## Potential issue with PyTorch

Cellpose and therefore the plugin and napari can crash without warning in some cases with torch==1.12.0. This can be fixed by reverting to an earlier version using:

    pip install torch==1.11.0


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
