# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from niftynet.layer import layer_util
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.downsample import DownSampleLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.crop import CropLayer
from niftynet.layer.linear_resize import LinearResizeLayer
from niftynet.utilities.util_common import look_up_operations


class UNet3D(TrainableLayer):
    """
    Implementation of No New-Net
      Isensee et al., "No New-Net", MICCAI BrainLesion Workshop 2018.
      The major changes between this and our standard 3d U-Net:
      * input size == output size: padded convs are used
      * leaky relu as non-linearity
      * reduced number of filters before upsampling
      * instance normalization (not batch)
      * fits 128x128x128 with batch size of 2 on one TitanX GPU for
      training
      * no learned upsampling: linear resizing. 
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='leakyrelu',
                 name='NoNewNet'):
        super(UNet3D, self).__init__(name=name)

        self.acti_func = acti_func
        self.num_classes = num_classes

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

        ## Number of convolution features at each stage
        self.n_features = [30, 60, 120, 240, 480]
        
        ## Pooling dimensions (should be len(self.n_features) - 1 long list)
        pooldims = [3, 3, 2, 2]
        assert len(pooldims) == len(self.n_features) - 1
        
        def pools_to_functions(pooldims):
            path_down = [str(i).replace('3', 'DOWNSAMPLE').replace('2', 'DOWNSAMPLE2D') for i in pooldims]
            path_up = [str(i).replace('3', 'UPSAMPLE').replace('2', 'UPSAMPLE2D') for i in reversed(pooldims)]
            return path_down + path_up + ['NONE']
        
        self.functions = pools_to_functions(pooldims)
        
        def nfeats_to_feats(nfeats, num_classes):
            path_down = [(f, f) for f in nfeats[:-1]]
            path_up = list(zip(reversed(nfeats), reversed(nfeats[:-1])))
            return path_down + path_up + [(nfeats[0], nfeats[0], num_classes)]
        
        self.feats = nfeats_to_feats(self.n_features, num_classes)
        
        def dims_to_names(dims):
            path_down = ['L{}-{}D'.format(i,j) for i,j in enumerate(dims)]
            path_up = ['R{}-{}D'.format(i,j) for i,j in reversed(list(enumerate(dims)))]
            return path_down + ['bottom'] + path_up
        
        self.names = dims_to_names(pooldims)
 
        self.kernels = [(3,3)]*(len(self.names)-1) + [(3,3,1)]
        self.with_downsample_branch = ['DOWNSAMPLE' in i for i in self.functions]
        
        print('using {}'.format(name))

    def layer_op(self, thru_tensor, is_training=True, keep_prob=0.5, **unused_kwargs):
        """
        :param thru_tensor: the input is modified in-place as it goes through the network
        :param is_training:
        :param unused_kwargs:
        :return:
        """
        # image_size  should be divisible by 16 because of max-pooling 4 times, 2x2x2
        
        #assert layer_util.check_spatial_dims(thru_tensor, lambda x: x % 16 == 0)
        convs = []
        for func, feat, name, kernel, wdsb in zip(
                self.functions, self.feats, self.names, 
                self.kernels, self.with_downsample_branch):
    
            block_layer = UNetBlock(func,
                                    feat,
                                    kernel, 
                                    with_downsample_branch=wdsb,
                                    w_initializer=self.initializers['w'],
                                    w_regularizer=self.regularizers['w'],
                                    acti_func=self.acti_func,
                                    name=name)

            if 'L' in name:
                thru_tensor, conv = block_layer(thru_tensor, is_training,
                                  keep_prob=keep_prob)
                convs.append(conv)
            elif 'R' in name:
                concat = ElementwiseLayer('CONCAT')(convs.pop(), thru_tensor)
                thru_tensor, _ = block_layer(concat, is_training,
                                  keep_prob=keep_prob)
            else:
                thru_tensor, _ = block_layer(thru_tensor, is_training,
                                keep_prob=keep_prob)
                
            print(block_layer)

        return thru_tensor


SUPPORTED_OP = {'DOWNSAMPLE', 'DOWNSAMPLE2D', 'UPSAMPLE', 'UPSAMPLE2D', 'NONE'}


class UNetBlock(TrainableLayer):
    def __init__(self,
                 func,
                 n_chns,
                 kernels,
                 w_initializer=None,
                 w_regularizer=None,
                 with_downsample_branch=False,
                 acti_func='leakyrelu',
                 name='UNet_block'):

        super(UNetBlock, self).__init__(name=name)

        self.func = look_up_operations(func.upper(), SUPPORTED_OP)

        self.kernels = kernels
        self.n_chns = n_chns
        self.with_downsample_branch = with_downsample_branch
        self.acti_func = acti_func

        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}
        
    def layer_op(self, thru_tensor, is_training, keep_prob=0.5):
        for (kernel_size, n_features) in zip(self.kernels, self.n_chns):
            # no activation or dropout after the final 1x1x1 conv layer
            if kernel_size == 1:
                acti_func = None
                feature_normalization = None
                keep_prob = None
            else:
                acti_func = self.acti_func
                feature_normalization = 'instance'

            conv_op = ConvolutionalLayer(n_output_chns=n_features,
                                         kernel_size=kernel_size,
                                         w_initializer=self.initializers['w'],
                                         w_regularizer=self.regularizers['w'],
                                         acti_func=acti_func,
                                         name='{}'.format(n_features),
                                         feature_normalization=feature_normalization)

            thru_tensor = conv_op(thru_tensor, is_training, keep_prob=keep_prob)

        if self.with_downsample_branch:
            branch_output = thru_tensor
        else:
            branch_output = None

        if self.func == 'DOWNSAMPLE':
            downsample_op = DownSampleLayer('MAX', kernel_size=2, stride=2, name='down_2x2')
            thru_tensor = downsample_op(thru_tensor)
        elif self.func == 'DOWNSAMPLE2D':
            downsample_op = DownSampleLayer('MAX', kernel_size=2, stride=2, name='down_2x2', dims=2)
            thru_tensor = downsample_op(thru_tensor)
        elif self.func == 'UPSAMPLE':
            up_shape = [2 * int(thru_tensor.shape[i]) for i in (1, 2, 3)]
            upsample_op = LinearResizeLayer(up_shape)
            thru_tensor = upsample_op(thru_tensor)
        elif self.func == 'UPSAMPLE2D':
            up_shape = [2 * int(thru_tensor.shape[i]) for i in (1, 2)] + [thru_tensor.shape[3]]
            upsample_op = LinearResizeLayer(up_shape)
            thru_tensor = upsample_op(thru_tensor)            

        elif self.func == 'NONE':
            pass  # do nothing
        return thru_tensor, branch_output
