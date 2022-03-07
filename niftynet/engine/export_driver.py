# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import tensorflow as tf

from niftynet.engine.handler_model import ModelRestorer
from niftynet.io.misc_io import infer_latest_model_file
from niftynet.utilities.util_common import \
    set_cuda_device, tf_config, device_string
from niftynet.engine.application_factory import ApplicationNetFactory
from niftynet.layer.post_processing import PostProcessingLayer

# pylint: disable=too-many-instance-attributes
class ExportDriver(object):
    """
    Class for exporting a model graph to be used with TF Serving.
    """

    def __init__(self):
        self.app = None
        self.model_dir = None
        self.vars_to_restore = ''
        self.initial_iter = 0
        self._event_handlers = None

    def initialise_application(self, workflow_param, data_param=None):
        try:
            system_param = workflow_param.get('SYSTEM', None)
            net_param = workflow_param.get('NETWORK', None)
            infer_param = workflow_param.get('INFERENCE', None)
            app_param = workflow_param.get('CUSTOM', None)
        except AttributeError:
            tf.compat.v1.logging.fatal('parameters should be dictionaries')
            raise

        assert os.path.exists(system_param.model_dir), \
            'Model folder not exists {}'.format(system_param.model_dir)
        self.model_dir = system_param.model_dir

        assert infer_param, 'inference parameters not specified'
        self.initial_iter = infer_param.inference_iter
        action_param = infer_param

        # infer the initial iteration from model files
        if self.initial_iter < 0:
            self.initial_iter = infer_latest_model_file(
                os.path.join(self.model_dir, 'models'))

        # create an application instance
        assert app_param, 'application specific param. not specified'
        self.app = GraphBuilder(net_param, action_param, system_param.action)

        self.app.initialise_dataset_loader(data_param, app_param)

    def run(self, application, graph=None):
        graph, in_tensor, out_tensor = ExportDriver.create_graph(application)
        restorer = ModelRestorer(model_dir=self.model_dir, initial_iter=self.initial_iter, is_training_action=False)
        save_dir = os.path.join(self.model_dir, 'saved-model')

        with tf.compat.v1.Session(config=tf_config(), graph=graph) as sess:
            restorer.restore_model(None)
            ExportDriver.backup_dir(save_dir)
            tf.saved_model.simple_save(sess, save_dir, {'input': in_tensor}, {'output': out_tensor})

    @staticmethod
    def backup_dir(path):
        if os.path.exists(path):
            backup_path = path + '-{}'
            i = 0
            while os.path.exists(backup_path.format(i)):
                i += 1
            os.rename(path, backup_path.format(i))

    # pylint: disable=not-context-manager
    @staticmethod
    def create_graph(application):
        graph = tf.Graph()
        with graph.as_default():
            application.initialise_network()
            with tf.name_scope('worker_0'):
                in_tensor, out_tensor = application.connect_data_and_network()
        return graph, in_tensor, out_tensor


class GraphBuilder():

    def __init__(self, net_param, action_param, action):
        self.net_param = net_param
        self.action_param = action_param
        self.data_param = None
        self.segmentation_param = None

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):

        self.data_param = data_param
        self.segmentation_param = task_param

    def initialise_network(self):
        self.net = ApplicationNetFactory.create(self.net_param.name)(
            num_classes=self.segmentation_param.num_classes,
            acti_func=self.net_param.activation_function)

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):

        input_section = self.segmentation_param.image[0]
        spatial_window_size = self.action_param.spatial_window_size or self.data_param[input_section].spatial_window_size
        if len(spatial_window_size) == 3 and spatial_window_size[2] == 1:
            spatial_window_size = spatial_window_size[:2]
        batch_size = None  # self.net_param.batch_size
        input_channels = 1
        input_shape = (batch_size,) + spatial_window_size + (input_channels,)

        net_in = tf.compat.v1.placeholder(tf.float32, shape=input_shape)
        net_args = {'is_training': False,
                    'keep_prob': self.net_param.keep_prob}
        net_out = self.net(net_in, **net_args)

        num_classes = self.segmentation_param.num_classes
        if num_classes > 1:
            post_process_layer = PostProcessingLayer(
                'ARGMAX', num_classes=num_classes)
        else:
            post_process_layer = PostProcessingLayer(
                'IDENTITY', num_classes=num_classes)
        net_out = post_process_layer(net_out)
        return net_in, net_out
