"""
Copyright (C) 2020 Christopher Brumann

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

from toolkit.utils import file as file_utils
from toolkit.importer.data_parser import parse_annotation, algorithm_parser
from toolkit.importer.argument_parser import ApplicationSettings
from toolkit.logger import logger
from toolkit.utils.configuration import infer_filenames_for_algorithm
from toolkit.utils.timer import timed

@timed
def numpy_converter(app_settings: ApplicationSettings, datasets):
    """ This program converts the algorithms output and annotation files into numpy arrays."""
    for dataset in datasets:
        dataset_name = dataset.dataset_name
        logger.info(f"Processing dataset '{dataset_name}'")
        npy_filename = file_utils.join_folders([dataset.npz_folder,
                                                f"{dataset_name}_annotation.npy"])
        if not file_utils.exists(npy_filename):
            logger.info(f"Converting annotation")
            annotation_data = parse_annotation(dataset.annotation_file)
            np.save(npy_filename, annotation_data.markers)
        else:
            logger.debug(f"Skipping {npy_filename}. File already exists")

        for algorithm in dataset.algorithms:
            algorithm_name = algorithm.algorithm_name
            npy_filename = file_utils.join_folders([dataset.npz_folder,
                                                    f"{dataset_name}_{algorithm_name}.npy"])

            if not file_utils.exists(npy_filename):
                logger.info(f"Converting {algorithm_name}")
                parser_fn = algorithm_parser[algorithm_name]["parser_function"]
                body_model = algorithm_parser[algorithm_name]["body_model"]

                detection_data = parser_fn(infer_filenames_for_algorithm(algorithm), body_model)
                np.save(npy_filename, detection_data.markers)
            else:
                logger.debug(f"Skipping {npy_filename}. File already exists")
