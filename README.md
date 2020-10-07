![logo](logo.png)
![version](https://img.shields.io/badge/version-v1.0.0-informational?style=for-the-badge)
![pythonverion](https://img.shields.io/badge/python-3.8-information?style=for-the-badge&logo=python)
## Feet Analysis Toolkit

This application provides a toolkit and scripts used in "Evaluation of pre-trained and open-source deep convolutional neural networks suitable for player detection and motion analysis in squash". 

## Getting started (Linux):

### Setup application

1. Create virtual environment

    `python3 -m venv /path/to/new/virtual/environment`

1. Activate the environment

    `source /path/to/new/virtual/environment/bin/activate`

1. Install requirements from `requirements.txt`

    `pip install -r requirements.txt`

### Setup data folder

1. Create a softlink for the dataset folder

    `ln -s ~/Projects/released-datasets/ ./dataset`

1. Create a clean output folder for results. 

    `mkdir output`

### Run

1. Run the application inside your virtual environment and provide dataset description file and your output folder

    `python run_toolkit.py --description=<DATASET_FOLDER>/dataset_description.json --output=./output`

### Other useful parameters (optional)
| Parameter | Description                                   |
|-----------|-----------------------------------------------|
| --debug   | Prints debugging log output during evaluation |
| --render  | Enables rendering while processing (**very slow**) |


### Citation
Please cite in your publications if it helps your research:
    
    TBA: BibTeX Entry  

TBA: [link to paper]()

### License
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.