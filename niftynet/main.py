# -*- coding: utf-8 -*-
"""

.. module:: niftynet
   :synopsis: Entry points for the NiftyNet CLI.

"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

# Before doing anything else, check TF is installed
# and fail gracefully if not.
try:
    import tensorflow as tf
except ImportError:
    raise ImportError('NiftyNet is based on TensorFlow, which'
                      ' does not seem to be installed on your'
                      ' system.\n\nPlease install TensorFlow'
                      ' (https://www.tensorflow.org/) to be'
                      ' able to use NiftyNet.')

try:
    from distutils.version import LooseVersion
    minimal_required_version = LooseVersion("1.5")
    tf_version = LooseVersion(tf.__version__)
    if tf_version < minimal_required_version:
        tf.compat.v1.logging.fatal('TensorFlow %s or later is required.'
                         '\n\nPlease upgrade TensorFlow'
                         ' (https://www.tensorflow.org/) to be'
                         ' able to use NiftyNet.\nCurrently using '
                         'TensorFlow %s:\ninstalled at %s\n\n',
                         minimal_required_version, tf_version, tf.__file__)
        raise ImportError
    else:
        tf.compat.v1.logging.info('TensorFlow version %s', tf_version)
except AttributeError:
    pass

import os

from niftynet.io.misc_io import set_logger, close_logger

set_logger()

from niftynet.utilities.util_import import require_module

require_module('blinker', descriptor='New dependency', mandatory=True)

from niftynet.engine.signal import TRAIN, INFER, EVAL, EXPORT
import niftynet.utilities.util_common as util
import niftynet.utilities.user_parameters_parser as user_parameters_parser
from niftynet.engine.application_driver import ApplicationDriver
from niftynet.engine.export_driver import ExportDriver
from niftynet.evaluation.evaluation_application_driver import \
    EvaluationApplicationDriver
from niftynet.io.misc_io import touch_folder, resolve_job_dir, to_absolute_path, copyfile_if_possible
import shutil

def main():
    system_param, input_data_param = user_parameters_parser.run()
    if util.has_bad_inputs(system_param):
        return -1

    # print all parameters to txt file for future reference
    all_param = {}
    all_param.update(system_param)
    all_param.update(input_data_param)

    # Set up path for niftynet model_root
    # (rewriting user input with an absolute path)
    system_param['SYSTEM'].model_dir = resolve_job_dir(
        system_param['SYSTEM'].job_dir,
        create_new=system_param['SYSTEM'].action == TRAIN)

    # copy config to output dir
    config_source = system_param['CONFIG_FILE'].path
    config_target = to_absolute_path(
                input_path=os.path.basename(config_source),
                root=system_param['SYSTEM'].model_dir)
    copyfile_if_possible(config_source, config_target)
    # writing all params for future reference
    txt_file = 'settings_{}.txt'.format(system_param['SYSTEM'].action)
    txt_file = os.path.join(system_param['SYSTEM'].model_dir, txt_file)
    try:
        util.print_save_input_parameters(all_param, txt_file)
    except IOError:
        tf.compat.v1.logging.fatal(
            'Unable to write %s,\nplease check '
            'model_dir parameter, current value: %s',
            txt_file, system_param['SYSTEM'].model_dir)
        raise

    # keep all commandline outputs to model_root
    log_file_name = os.path.join(
        system_param['SYSTEM'].model_dir,
        '{}_{}'.format(all_param['SYSTEM'].action, 'niftynet_log'))
    set_logger(file_name=log_file_name)

    # set up all model folder related parameters here
    # see https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/issues/168
    # 1. resolve mapping file:
    try:
        if system_param['NETWORK'].histogram_ref_file:
            system_param['NETWORK'].histogram_ref_file = to_absolute_path(
                input_path=system_param['NETWORK'].histogram_ref_file,
                root=system_param['SYSTEM'].model_dir)
    except (AttributeError, KeyError):
        pass
    # 2. resolve output file:
    try:
        if system_param['INFERENCE'].save_seg_dir:
            system_param['INFERENCE'].save_seg_dir = to_absolute_path(
                input_path=system_param['INFERENCE'].save_seg_dir,
                root=system_param['SYSTEM'].model_dir)
    except (AttributeError, KeyError):
        pass
    # 3. resolve dataset splitting file:
    try:
        if system_param['SYSTEM'].dataset_split_file:
            split_file_source = to_absolute_path(
                input_path=system_param['SYSTEM'].dataset_split_file,
                root=os.path.dirname(system_param['CONFIG_FILE'].path))
            split_file_name = os.path.basename(split_file_source)
            split_file_target = to_absolute_path(split_file_name,
                root=system_param['SYSTEM'].model_dir)
            copyfile_if_possible(split_file_source, split_file_target)
            system_param['SYSTEM'].dataset_split_file = split_file_target
    except (AttributeError, KeyError):
        pass

    # 4. resolve evaluation dir:
    try:
        if system_param['EVALUATION'].save_csv_dir:
            system_param['EVALUATION'].save_csv_dir = to_absolute_path(
                input_path=system_param['EVALUATION'].save_csv_dir,
                root=system_param['SYSTEM'].model_dir)
    except (AttributeError, KeyError):
        pass

    # start application
    driver_table = {
        TRAIN: ApplicationDriver,
        INFER: ApplicationDriver,
        EXPORT: ExportDriver,
        EVAL: EvaluationApplicationDriver}
    app_driver = driver_table[system_param['SYSTEM'].action]()
    app_driver.initialise_application(system_param, input_data_param)
    app_driver.run(app_driver.app)

    if tf.compat.v1.get_default_session() is not None:
        tf.compat.v1.get_default_session().close()
    tf.compat.v1.reset_default_graph()
    close_logger()

    return 0

if __name__ == '__main__':
    main()
