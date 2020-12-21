

import os
import random
import pickle
import numpy as np
from typing import List
import tensorflow as tf
from typing import Optional
import tensorflow_addons as tfa
from tensorflow.keras import losses
from tensorflow.keras import layers
from tensorflow.keras import backend
import tensorflow_probability as tfp
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow_addons.activations import sparsemax

P_MIN = 0.0005
P_MAX = 0.9995

p_min = 0.0005
p_max = 0.9995

def logloss(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred,p_min,p_max)
    return -backend.mean(y_true*backend.log(y_pred) + (1-y_true)*backend.log(1-y_pred))

def weighted_logloss(weights):
    def wloss(y_true, y_pred):
        k_weights = backend.constant(weights)
        y_true = backend.cast(y_true, y_pred.dtype)
        return logloss(y_true, y_pred) * backend.sum(y_true * k_weights, axis=-1)
    return wloss

def create_tabnet(n_features, n_features_2, n_labels, label_smoothing = 0.0005, brute_force=False, weights=None):
    def register_keras_custom_object(cls):
        tf.keras.utils.get_custom_objects()[cls.__name__] = cls
        return cls

    def glu(x, n_units=None):
        if n_units is None: n_units = tf.shape(x)[-1] // 2
        return x[..., :n_units] * tf.nn.sigmoid(x[..., n_units:])

    @register_keras_custom_object
    @tf.function
    def sparsemax(logits, axis):
        logits = tf.convert_to_tensor(logits, name="logits")
        shape = logits.get_shape()
        rank = shape.rank
        is_last_axis = (axis == -1) or (axis == rank - 1)
        if is_last_axis:
            output = _compute_2d_sparsemax(logits)
            output.set_shape(shape)
            return output
        rank_op = tf.rank(logits)
        axis_norm = axis % rank
        logits = _swap_axis(logits, axis_norm, tf.math.subtract(rank_op, 1))
        output = _compute_2d_sparsemax(logits)
        output = _swap_axis(output, axis_norm, tf.math.subtract(rank_op, 1))
        output.set_shape(shape)
        return output

    def _swap_axis(logits, dim_index, last_index, **kwargs):
        return tf.transpose(
            logits,
            tf.concat(
                [
                    tf.range(dim_index),
                    [last_index],
                    tf.range(dim_index + 1, last_index),
                    [dim_index],
                ],
                0,
            ),
            **kwargs,
        )

    def _compute_2d_sparsemax(logits):
        shape_op = tf.shape(logits)
        obs = tf.math.reduce_prod(shape_op[:-1])
        dims = shape_op[-1]
        z = tf.reshape(logits, [obs, dims])
        z_sorted, _ = tf.nn.top_k(z, k=dims)
        z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
        k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
        z_check = 1 + k * z_sorted > z_cumsum
        k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)
        k_z_safe = tf.math.maximum(k_z, 1)
        indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
        tau_sum = tf.gather_nd(z_cumsum, indices)
        tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)
        p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))
        p_safe = tf.where(
            tf.expand_dims(
                tf.math.logical_or(tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),
                axis=-1,
            ),
            tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
            p,
        )
        p_safe = tf.reshape(p_safe, shape_op)
        return p_safe

    class GhostBatchNormalization(tf.keras.Model):
        def __init__(
                self, virtual_divider: int = 1, momentum: float = 0.9, epsilon: float = 1e-5
        ):
            super(GhostBatchNormalization, self).__init__()
            self.virtual_divider = virtual_divider
            self.bn = BatchNormInferenceWeighting(momentum=momentum)

        def call(self, x, training: bool = None, alpha: float = 0.0):
            if training:
                chunks = tf.split(x, self.virtual_divider)
                x = [self.bn(x, training=True) for x in chunks]
                return tf.concat(x, 0)
            return self.bn(x, training=False, alpha=alpha)

        @property
        def moving_mean(self):
            return self.bn.moving_mean

        @property
        def moving_variance(self):
            return self.bn.moving_variance

    class BatchNormInferenceWeighting(tf.keras.layers.Layer):
        def __init__(self, momentum: float = 0.9, epsilon: float = None):
            super(BatchNormInferenceWeighting, self).__init__()
            self.momentum = momentum
            self.epsilon = tf.keras.backend.epsilon() if epsilon is None else epsilon

        def build(self, input_shape):
            channels = input_shape[-1]

            self.gamma = tf.Variable(
                initial_value=tf.ones((channels,), tf.float32), trainable=True,
            )
            self.beta = tf.Variable(
                initial_value=tf.zeros((channels,), tf.float32), trainable=True,
            )

            self.moving_mean = tf.Variable(
                initial_value=tf.zeros((channels,), tf.float32), trainable=False,
            )
            self.moving_mean_of_squares = tf.Variable(
                initial_value=tf.zeros((channels,), tf.float32), trainable=False,
            )

        def __update_moving(self, var, value):
            var.assign(var * self.momentum + (1 - self.momentum) * value)

        def __apply_normalization(self, x, mean, variance):
            return self.gamma * (x - mean) / tf.sqrt(variance + self.epsilon) + self.beta

        def call(self, x, training: bool = None, alpha: float = 0.0):
            mean = tf.reduce_mean(x, axis=0)
            mean_of_squares = tf.reduce_mean(tf.pow(x, 2), axis=0)

            if training:
                # update moving stats
                self.__update_moving(self.moving_mean, mean)
                self.__update_moving(self.moving_mean_of_squares, mean_of_squares)

                variance = mean_of_squares - tf.pow(mean, 2)
                x = self.__apply_normalization(x, mean, variance)
            else:
                mean = alpha * mean + (1 - alpha) * self.moving_mean
                variance = (
                                   alpha * mean_of_squares + (1 - alpha) * self.moving_mean_of_squares
                           ) - tf.pow(mean, 2)
                x = self.__apply_normalization(x, mean, variance)

            return x

    class FeatureBlock(tf.keras.Model):
        def __init__(
                self,
                feature_dim: int,
                apply_glu: bool = True,
                bn_momentum: float = 0.9,
                bn_virtual_divider: int = 32,
                fc: tf.keras.layers.Layer = None,
                epsilon: float = 1e-5,
        ):
            super(FeatureBlock, self).__init__()
            self.apply_gpu = apply_glu
            self.feature_dim = feature_dim
            units = feature_dim * 2 if apply_glu else feature_dim

            self.fc = tf.keras.layers.Dense(units, use_bias=False) if fc is None else fc
            self.bn = GhostBatchNormalization(
                virtual_divider=bn_virtual_divider, momentum=bn_momentum
            )

        def call(self, x, training: bool = None, alpha: float = 0.0):
            x = self.fc(x)
            x = self.bn(x, training=training, alpha=alpha)
            if self.apply_gpu:
                return glu(x, self.feature_dim)
            return x

    class AttentiveTransformer(tf.keras.Model):
        def __init__(self, feature_dim: int, bn_momentum: float, bn_virtual_divider: int):
            super(AttentiveTransformer, self).__init__()
            self.block = FeatureBlock(
                feature_dim,
                bn_momentum=bn_momentum,
                bn_virtual_divider=bn_virtual_divider,
                apply_glu=False,
            )

        def call(self, x, prior_scales, training=None, alpha: float = 0.0):
            x = self.block(x, training=training, alpha=alpha)
            return sparsemax(x * prior_scales, -1)

    class FeatureTransformer(tf.keras.Model):
        def __init__(
                self,
                feature_dim: int,
                fcs: List[tf.keras.layers.Layer] = [],
                n_total: int = 4,
                n_shared: int = 2,
                bn_momentum: float = 0.9,
                bn_virtual_divider: int = 1,
        ):
            super(FeatureTransformer, self).__init__()
            self.n_total, self.n_shared = n_total, n_shared

            kargs = {
                "feature_dim": feature_dim,
                "bn_momentum": bn_momentum,
                "bn_virtual_divider": bn_virtual_divider,
            }

            # build blocks
            self.blocks: List[FeatureBlock] = []
            for n in range(n_total):
                # some shared blocks
                if fcs and n < len(fcs):
                    self.blocks.append(FeatureBlock(**kargs, fc=fcs[n]))
                # build new blocks
                else:
                    self.blocks.append(FeatureBlock(**kargs))

        def call(
                self, x: tf.Tensor, training: bool = None, alpha: float = 0.0
        ) -> tf.Tensor:
            x = self.blocks[0](x, training=training, alpha=alpha)
            for n in range(1, self.n_total):
                x = x * tf.sqrt(0.5) + self.blocks[n](x, training=training, alpha=alpha)
            return x

        @property
        def shared_fcs(self):
            return [self.blocks[i].fc for i in range(self.n_shared)]

    @register_keras_custom_object
    class GroupNormalization(tf.keras.layers.Layer):

        def __init__(
                self,
                groups: int = 2,
                axis: int = -1,
                epsilon: float = 1e-3,
                center: bool = True,
                scale: bool = True,
                beta_initializer="zeros",
                gamma_initializer="ones",
                beta_regularizer=None,
                gamma_regularizer=None,
                beta_constraint=None,
                gamma_constraint=None,
                **kwargs
        ):
            super().__init__(**kwargs)
            self.supports_masking = True
            self.groups = groups
            self.axis = axis
            self.epsilon = epsilon
            self.center = center
            self.scale = scale
            self.beta_initializer = tf.keras.initializers.get(beta_initializer)
            self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
            self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
            self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
            self.beta_constraint = tf.keras.constraints.get(beta_constraint)
            self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
            self._check_axis()

        def build(self, input_shape):

            self._check_if_input_shape_is_none(input_shape)
            self._set_number_of_groups_for_instance_norm(input_shape)
            self._check_size_of_dimensions(input_shape)
            self._create_input_spec(input_shape)

            self._add_gamma_weight(input_shape)
            self._add_beta_weight(input_shape)
            self.built = True
            super().build(input_shape)

        def call(self, inputs, training=None):
            # Training=none is just for compat with batchnorm signature call
            input_shape = tf.keras.backend.int_shape(inputs)
            tensor_input_shape = tf.shape(inputs)

            reshaped_inputs, group_shape = self._reshape_into_groups(
                inputs, input_shape, tensor_input_shape
            )

            normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

            outputs = tf.reshape(normalized_inputs, tensor_input_shape)

            return outputs

        def get_config(self):
            config = {
                "groups": self.groups,
                "axis": self.axis,
                "epsilon": self.epsilon,
                "center": self.center,
                "scale": self.scale,
                "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
                "gamma_initializer": tf.keras.initializers.serialize(
                    self.gamma_initializer
                ),
                "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
                "gamma_regularizer": tf.keras.regularizers.serialize(
                    self.gamma_regularizer
                ),
                "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
                "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
            }
            base_config = super().get_config()
            return {**base_config, **config}

        def compute_output_shape(self, input_shape):
            return input_shape

        def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

            group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape

        def _apply_normalization(self, reshaped_inputs, input_shape):

            group_shape = tf.keras.backend.int_shape(reshaped_inputs)
            group_reduction_axes = list(range(1, len(group_shape)))
            axis = -2 if self.axis == -1 else self.axis - 1
            group_reduction_axes.pop(axis)

            mean, variance = tf.nn.moments(
                reshaped_inputs, group_reduction_axes, keepdims=True
            )

            gamma, beta = self._get_reshaped_weights(input_shape)
            normalized_inputs = tf.nn.batch_normalization(
                reshaped_inputs,
                mean=mean,
                variance=variance,
                scale=gamma,
                offset=beta,
                variance_epsilon=self.epsilon,
            )
            return normalized_inputs

        def _get_reshaped_weights(self, input_shape):
            broadcast_shape = self._create_broadcast_shape(input_shape)
            gamma = None
            beta = None
            if self.scale:
                gamma = tf.reshape(self.gamma, broadcast_shape)

            if self.center:
                beta = tf.reshape(self.beta, broadcast_shape)
            return gamma, beta

        def _check_if_input_shape_is_none(self, input_shape):
            dim = input_shape[self.axis]
            if dim is None:
                raise ValueError(
                    "Axis " + str(self.axis) + " of "
                                               "input tensor should have a defined dimension "
                                               "but the layer received an input with shape " + str(input_shape) + "."
                )

        def _set_number_of_groups_for_instance_norm(self, input_shape):
            dim = input_shape[self.axis]

            if self.groups == -1:
                self.groups = dim

        def _check_size_of_dimensions(self, input_shape):

            dim = input_shape[self.axis]
            if dim < self.groups:
                raise ValueError(
                    "Number of groups (" + str(self.groups) + ") cannot be "
                                                              "more than the number of channels (" + str(dim) + ")."
                )

            if dim % self.groups != 0:
                raise ValueError(
                    "Number of groups (" + str(self.groups) + ") must be a "
                                                              "multiple of the number of channels (" + str(dim) + ")."
                )

        def _check_axis(self):

            if self.axis == 0:
                raise ValueError(
                    "You are trying to normalize your batch axis. Do you want to "
                    "use tf.layer.batch_normalization instead"
                )

        def _create_input_spec(self, input_shape):

            dim = input_shape[self.axis]
            self.input_spec = tf.keras.layers.InputSpec(
                ndim=len(input_shape), axes={self.axis: dim}
            )

        def _add_gamma_weight(self, input_shape):

            dim = input_shape[self.axis]
            shape = (dim,)

            if self.scale:
                self.gamma = self.add_weight(
                    shape=shape,
                    name="gamma",
                    initializer=self.gamma_initializer,
                    regularizer=self.gamma_regularizer,
                    constraint=self.gamma_constraint,
                )
            else:
                self.gamma = None

        def _add_beta_weight(self, input_shape):

            dim = input_shape[self.axis]
            shape = (dim,)

            if self.center:
                self.beta = self.add_weight(
                    shape=shape,
                    name="beta",
                    initializer=self.beta_initializer,
                    regularizer=self.beta_regularizer,
                    constraint=self.beta_constraint,
                )
            else:
                self.beta = None

        def _create_broadcast_shape(self, input_shape):
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
            return broadcast_shape

    class TransformBlock(tf.keras.Model):

        def __init__(self, features,
                     norm_type,
                     momentum=0.9,
                     virtual_batch_size=None,
                     groups=2,
                     block_name='',
                     **kwargs):
            super(TransformBlock, self).__init__(**kwargs)
            self.features = features
            self.norm_type = norm_type
            self.momentum = momentum
            self.groups = groups
            self.virtual_batch_size = virtual_batch_size
            self.transform = tf.keras.layers.Dense(self.features, use_bias=False,
                                                   name=f'transformblock_dense_{block_name}')
            if norm_type == 'batch':
                self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=momentum,
                                                             virtual_batch_size=virtual_batch_size,
                                                             name=f'transformblock_bn_{block_name}')
            else:
                self.bn = GroupNormalization(axis=-1, groups=self.groups, name=f'transformblock_gn_{block_name}')

        def call(self, inputs, training=None):
            x = self.transform(inputs)
            x = self.bn(x, training=training)
            return x

    class TabNet(tf.keras.Model):
        def __init__(
                self,
                num_features: int,
                feature_dim: int,
                output_dim: int,
                feature_columns: List = None,
                num_decision_steps: int = 1,
                n_total: int = 4,
                n_shared: int = 2,
                relaxation_factor: float = 1.5,
                bn_epsilon: float = 1e-5,
                batch_momentum: float = 0.7,
                bn_virtual_divider: int = 1,
                sparsity_coefficient=0,
        ):
            super(TabNet, self).__init__()
            self.output_dim, self.num_features = output_dim, num_features
            self.n_step, self.relaxation_factor = num_decision_steps, relaxation_factor
            self.feature_columns = feature_columns
            if feature_columns is not None:
                self.input_features = tf.keras.layers.DenseFeatures(feature_columns)
            self.bn = tf.keras.layers.BatchNormalization(
                momentum=batch_momentum, epsilon=bn_epsilon
            )
            kargs = {
                "feature_dim": feature_dim + output_dim,
                "n_total": n_total,
                "n_shared": n_shared,
                "bn_momentum": batch_momentum,
                "bn_virtual_divider": bn_virtual_divider,
            }
            self.feature_transforms = [FeatureTransformer(**kargs)]
            self.attentive_transforms = []
            for i in range(num_decision_steps):
                self.feature_transforms.append(FeatureTransformer(**kargs, fcs=self.feature_transforms[0].shared_fcs))
                self.attentive_transforms.append(AttentiveTransformer(num_features, batch_momentum, bn_virtual_divider))

        def call(self, features, training=None, alpha=0.0):
            if self.feature_columns is not None:
                features = self.input_features(features)
            bs = tf.shape(features)[0]
            out_agg = tf.zeros((bs, self.output_dim))
            prior_scales = tf.ones((bs, self.num_features))
            masks = []
            features = self.bn(features, training=training)
            masked_features = features
            total_entropy = 0.0
            for step_i in range(self.n_step + 1):
                x = self.feature_transforms[step_i](
                    masked_features, training=training, alpha=alpha
                )
                if step_i > 0:
                    out = tf.keras.activations.relu(x[:, : self.output_dim])
                    out_agg += out
                if step_i < self.n_step:
                    x_for_mask = x[:, self.output_dim:]
                    mask_values = self.attentive_transforms[step_i](
                        x_for_mask, prior_scales, training=training, alpha=alpha
                    )
                    prior_scales *= self.relaxation_factor - mask_values
                    masked_features = tf.multiply(mask_values, features)
                    total_entropy = tf.reduce_mean(
                        tf.reduce_sum(
                            tf.multiply(mask_values, tf.math.log(mask_values + 1e-15)),
                            axis=1,
                        )
                    )
                    masks.append(tf.expand_dims(tf.expand_dims(mask_values, 0), 3))
            loss = total_entropy / self.n_step
            return out_agg  # , loss, masks

    class TabNetClassifier(tf.keras.Model):
        def __init__(
                self,
                num_features: int,
                feature_dim: int,
                output_dim: int,
                n_classes: int,
                feature_columns: List = None,
                n_step: int = 1,
                n_total: int = 4,
                n_shared: int = 2,
                relaxation_factor: float = 1.5,
                sparsity_coefficient: float = 1e-5,
                bn_epsilon: float = 1e-5,
                bn_momentum: float = 0.7,
                bn_virtual_divider: int = 32,
                dp: float = None,
                **kwargs
        ):
            super(TabNetClassifier, self).__init__()

            self.configs = {
                "num_features": num_features,
                "feature_dim": feature_dim,
                "output_dim": output_dim,
                "n_classes": n_classes,
                "feature_columns": feature_columns,
                "n_step": n_step,
                "n_total": n_total,
                "n_shared": n_shared,
                "relaxation_factor": relaxation_factor,
                "sparsity_coefficient": sparsity_coefficient,
                "bn_epsilon": bn_epsilon,
                "bn_momentum": bn_momentum,
                "bn_virtual_divider": bn_virtual_divider,
                "dp": dp,
            }
            for k, v in kwargs.items():
                self.configs[k] = v

            self.sparsity_coefficient = sparsity_coefficient

            self.model = TabNet(
                feature_columns=feature_columns,
                num_features=num_features,
                feature_dim=feature_dim,
                output_dim=output_dim,
                num_decision_steps=n_step,
                relaxation_factor=relaxation_factor,
                bn_epsilon=bn_epsilon,
                batch_momentum=bn_momentum,
                bn_virtual_divider=bn_virtual_divider,
            )
            self.dp = tf.keras.layers.Dropout(dp) if dp is not None else dp
            self.head = tf.keras.layers.Dense(n_classes, activation=None, use_bias=False)

        def call(self, x, training: bool = None, alpha: float = 0.0):
            out, sparse_loss, _ = self.model(x, training=training, alpha=alpha)
            if self.dp is not None:
                out = self.dp(out, training=training)
            y = self.head(out, training=training)

            if training:
                self.add_loss(-self.sparsity_coefficient * sparse_loss)

            return y

        def get_config(self):
            return self.configs

        def save_to_directory(self, path_to_folder):
            self.save_weights(os.path.join(path_to_folder, "ckpt"), overwrite=True)
            with open(os.path.join(path_to_folder, "configs.pickle"), "wb") as f:
                pickle.dump(self.configs, f)

        @classmethod
        def load_from_directory(cls, path_to_folder):
            with open(os.path.join(path_to_folder, "configs.pickle"), "rb") as f:
                configs = pickle.load(f)
            model: tf.keras.Model = cls(**configs)
            model.build((None, configs["num_features"]))
            load_status = model.load_weights(os.path.join(path_to_folder, "ckpt"))
            load_status.expect_partial()
            return model

    class TabNetRegressor(tf.keras.Model):

        def __init__(self, feature_columns,
                     num_regressors,
                     num_features=None,
                     feature_dim=64,
                     output_dim=64,
                     num_decision_steps=5,
                     relaxation_factor=1.5,
                     sparsity_coefficient=1e-5,
                     norm_type='group',
                     batch_momentum=0.98,
                     virtual_batch_size=None,
                     num_groups=1,
                     epsilon=1e-5,
                     **kwargs):
            super(TabNetRegressor, self).__init__(**kwargs)
            self.num_regressors = num_regressors
            self.tabnet = TabNet(feature_columns=feature_columns,
                                 num_features=num_features,
                                 feature_dim=feature_dim,
                                 output_dim=output_dim,
                                 num_decision_steps=num_decision_steps,
                                 relaxation_factor=relaxation_factor,
                                 sparsity_coefficient=sparsity_coefficient,
                                 norm_type=norm_type,
                                 batch_momentum=batch_momentum,
                                 virtual_batch_size=virtual_batch_size,
                                 num_groups=num_groups,
                                 epsilon=epsilon,
                                 **kwargs)
            self.regressor = tf.keras.layers.Dense(num_regressors, use_bias=False, name='regressor')

        def call(self, inputs, training=None):
            self.activations = self.tabnet(inputs, training=training)
            out = self.regressor(self.activations)
            return out

        def summary(self, *super_args, **super_kwargs):
            super().summary(*super_args, **super_kwargs)
            self.tabnet.summary(*super_args, **super_kwargs)

    TabNetClassification = TabNetClassifier
    TabNetRegression = TabNetRegressor

    class StackedTabNet(tf.keras.Model):

        def __init__(self, feature_columns,
                     num_layers=1,
                     feature_dim=64,
                     output_dim=64,
                     num_features=None,
                     num_decision_steps=5,
                     relaxation_factor=1.5,
                     sparsity_coefficient=1e-5,
                     norm_type='group',
                     batch_momentum=0.98,
                     virtual_batch_size=None,
                     num_groups=2,
                     epsilon=1e-5,
                     **kwargs):
            super(StackedTabNet, self).__init__(**kwargs)
            if num_layers < 1: raise ValueError("`num_layers` cannot be less than 1")
            if type(feature_dim) not in [list, tuple]: feature_dim = [feature_dim] * num_layers
            if type(output_dim) not in [list, tuple]: output_dim = [output_dim] * num_layers
            if len(feature_dim) != num_layers: raise ValueError("`feature_dim` must be a list of length `num_layers`")
            if len(output_dim) != num_layers: raise ValueError("`output_dim` must be a list of length `num_layers`")
            self.num_layers = num_layers
            layers = []
            layers.append(TabNet(feature_columns=feature_columns,
                                 num_features=num_features,
                                 feature_dim=feature_dim[0],
                                 output_dim=output_dim[0],
                                 num_decision_steps=num_decision_steps,
                                 relaxation_factor=relaxation_factor,
                                 sparsity_coefficient=sparsity_coefficient,
                                 # norm_type=norm_type,
                                 batch_momentum=batch_momentum))
            # virtual_batch_size=virtual_batch_size,
            # num_groups=num_groups,
            # epsilon=epsilon))

            for layer_idx in range(1, num_layers):
                layers.append(TabNet(feature_columns=None,
                                     num_features=output_dim[layer_idx - 1],
                                     feature_dim=feature_dim[layer_idx],
                                     output_dim=output_dim[layer_idx],
                                     num_decision_steps=num_decision_steps,
                                     relaxation_factor=relaxation_factor,
                                     sparsity_coefficient=sparsity_coefficient,
                                     # norm_type=norm_type,
                                     batch_momentum=batch_momentum))
                # virtual_batch_size=virtual_batch_size,
                # num_groups=num_groups,
                # epsilon=epsilon))
            self.tabnet_layers = layers

        def call(self, inputs, training=None):
            x = self.tabnet_layers[0](inputs, training=training)
            for layer_idx in range(1, self.num_layers): x = self.tabnet_layers[layer_idx](x, training=training)
            return x

        @property
        def tabnets(self):
            return self.tabnet_layers

        @property
        def feature_selection_masks(self):
            return [tabnet.feature_selection_masks for tabnet in self.tabnet_layers]

        @property
        def aggregate_feature_selection_mask(self):
            return [tabnet.aggregate_feature_selection_mask for tabnet in self.tabnet_layers]

    class StackedTabNetClassifier(tf.keras.Model):

        def __init__(self, feature_columns,
                     num_classes,
                     num_layers=1,
                     feature_dim=64,
                     output_dim=64,
                     num_features=None,
                     num_decision_steps=5,
                     relaxation_factor=1.5,
                     sparsity_coefficient=1e-5,
                     norm_type='group',
                     batch_momentum=0.98,
                     virtual_batch_size=None,
                     num_groups=2,
                     epsilon=1e-5,
                     multi_label=False,
                     **kwargs):
            super(StackedTabNetClassifier, self).__init__(**kwargs)
            self.num_classes = num_classes
            self.stacked_tabnet = StackedTabNet(feature_columns=feature_columns,
                                                num_layers=num_layers,
                                                feature_dim=feature_dim,
                                                output_dim=output_dim,
                                                num_features=num_features,
                                                num_decision_steps=num_decision_steps,
                                                relaxation_factor=relaxation_factor,
                                                sparsity_coefficient=sparsity_coefficient,
                                                norm_type=norm_type,
                                                batch_momentum=batch_momentum,
                                                virtual_batch_size=virtual_batch_size,
                                                num_groups=num_groups,
                                                epsilon=epsilon)
            if multi_label:
                self.clf = tf.keras.layers.Dense(num_classes, activation='sigmoid', use_bias=False)
            else:
                self.clf = tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False)

        def call(self, inputs, training=None):
            self.activations = self.stacked_tabnet(inputs[0], training=training)
            acts = tf.keras.layers.Concatenate()([self.activations, inputs[1]])
            out = self.clf(acts)
            return out

    class StackedTabNetRegressor(tf.keras.Model):

        def __init__(self, feature_columns,
                     num_regressors,
                     num_layers=1,
                     feature_dim=64,
                     output_dim=64,
                     num_features=None,
                     num_decision_steps=5,
                     relaxation_factor=1.5,
                     sparsity_coefficient=1e-5,
                     norm_type='group',
                     batch_momentum=0.98,
                     virtual_batch_size=None,
                     num_groups=2,
                     epsilon=1e-5,
                     **kwargs):
            super(StackedTabNetRegressor, self).__init__(**kwargs)
            self.num_regressors = num_regressors
            self.stacked_tabnet = StackedTabNet(feature_columns=feature_columns,
                                                num_layers=num_layers,
                                                feature_dim=feature_dim,
                                                output_dim=output_dim,
                                                num_features=num_features,
                                                num_decision_steps=num_decision_steps,
                                                relaxation_factor=relaxation_factor,
                                                sparsity_coefficient=sparsity_coefficient,
                                                norm_type=norm_type,
                                                batch_momentum=batch_momentum,
                                                virtual_batch_size=virtual_batch_size,
                                                num_groups=num_groups,
                                                epsilon=epsilon)
            self.regressor = tf.keras.layers.Dense(num_regressors, use_bias=False)

        def call(self, inputs, training=None):
            self.activations = self.tabnet(inputs, training=training)
            out = self.regressor(self.activations)
            return out


    num_layers = 1
    feature_dim = 988
    output_dim = 6
    num_decision_steps = 5
    relax = 1.379977210038878
    sparse = 0.9919300062604508

    model = StackedTabNetClassifier(feature_columns=None, num_classes=206, num_layers=num_layers,
                                    feature_dim=feature_dim, output_dim=output_dim, num_features=n_features,
                                    num_decision_steps=num_decision_steps, relaxation_factor=relax,
                                    sparsity_coefficient=sparse, batch_momentum=0.98,
                                    virtual_batch_size=None, norm_type='group',
                                    num_groups=-1, multi_label=True)
    model.compile(optimizer=tfa.optimizers.SWA(tf.optimizers.Adam(lr=0.001), start_averaging=9, average_period=6), loss=weighted_logloss(weights), metrics=logloss)

    input_1_1 = layers.Input(shape = (n_features,), name = 'Input1')
    input_2_1 = layers.Input(shape = (n_features_2,), name = 'Input2')
    mask_1 = layers.Dropout(0.3)((tf.ones_like(input_1_1)))
    mask_2 = layers.Dropout(0.3)((tf.ones_like(input_2_1)))

    encoded = model([input_1_1 * mask_1, input_2_1 * mask_2])

    head_4 = Sequential([
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(256, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(n_features, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        layers.Dense(n_features, activation="sigmoid")
        ],name='Head4')
    decoded = head_4(encoded)
    reconstruction_1 = decoded * (1 - mask_1) + input_1_1 * mask_1

    head_5 = Sequential([
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(256, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(n_features_2, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        layers.Dense(n_features_2, activation="sigmoid")
        ],name='Head5')
    decoded = head_5(encoded)
    reconstruction_2 = decoded * (1 - mask_2) + input_2_1 * mask_2

    loss = tf.reduce_mean(tf.math.abs(input_1_1 - reconstruction_1)) + tf.reduce_mean(tf.math.abs(input_2_1 - reconstruction_2))
    ae_model = Model(inputs=[input_1_1, input_2_1], outputs=[loss])
    ae_model.compile(optimizer=tf.optimizers.Adam(), loss=lambda t, y: y)

    return model, ae_model

def create_tabnet_2(n_features, n_features_2, n_labels, label_smoothing = 0.0005, brute_force=False, weights=None):
    def register_keras_custom_object(cls):
        tf.keras.utils.get_custom_objects()[cls.__name__] = cls
        return cls

    def glu(x, n_units=None):
        if n_units is None: n_units = tf.shape(x)[-1] // 2
        return x[..., :n_units] * tf.nn.sigmoid(x[..., n_units:])

    @register_keras_custom_object
    @tf.function
    def sparsemax(logits, axis):
        logits = tf.convert_to_tensor(logits, name="logits")
        shape = logits.get_shape()
        rank = shape.rank
        is_last_axis = (axis == -1) or (axis == rank - 1)
        if is_last_axis:
            output = _compute_2d_sparsemax(logits)
            output.set_shape(shape)
            return output
        rank_op = tf.rank(logits)
        axis_norm = axis % rank
        logits = _swap_axis(logits, axis_norm, tf.math.subtract(rank_op, 1))
        output = _compute_2d_sparsemax(logits)
        output = _swap_axis(output, axis_norm, tf.math.subtract(rank_op, 1))
        output.set_shape(shape)
        return output

    def _swap_axis(logits, dim_index, last_index, **kwargs):
        return tf.transpose(
            logits,
            tf.concat(
                [
                    tf.range(dim_index),
                    [last_index],
                    tf.range(dim_index + 1, last_index),
                    [dim_index],
                ],
                0,
            ),
            **kwargs,
        )

    def _compute_2d_sparsemax(logits):
        shape_op = tf.shape(logits)
        obs = tf.math.reduce_prod(shape_op[:-1])
        dims = shape_op[-1]
        z = tf.reshape(logits, [obs, dims])
        z_sorted, _ = tf.nn.top_k(z, k=dims)
        z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
        k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
        z_check = 1 + k * z_sorted > z_cumsum
        k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)
        k_z_safe = tf.math.maximum(k_z, 1)
        indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
        tau_sum = tf.gather_nd(z_cumsum, indices)
        tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)
        p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))
        p_safe = tf.where(
            tf.expand_dims(
                tf.math.logical_or(tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),
                axis=-1,
            ),
            tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
            p,
        )
        p_safe = tf.reshape(p_safe, shape_op)
        return p_safe

    @register_keras_custom_object
    class GroupNormalization(tf.keras.layers.Layer):

        def __init__(
                self,
                groups: int = 2,
                axis: int = -1,
                epsilon: float = 1e-3,
                center: bool = True,
                scale: bool = True,
                beta_initializer="zeros",
                gamma_initializer="ones",
                beta_regularizer=None,
                gamma_regularizer=None,
                beta_constraint=None,
                gamma_constraint=None,
                **kwargs
        ):
            super().__init__(**kwargs)
            self.supports_masking = True
            self.groups = groups
            self.axis = axis
            self.epsilon = epsilon
            self.center = center
            self.scale = scale
            self.beta_initializer = tf.keras.initializers.get(beta_initializer)
            self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
            self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
            self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
            self.beta_constraint = tf.keras.constraints.get(beta_constraint)
            self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
            self._check_axis()

        def build(self, input_shape):

            self._check_if_input_shape_is_none(input_shape)
            self._set_number_of_groups_for_instance_norm(input_shape)
            self._check_size_of_dimensions(input_shape)
            self._create_input_spec(input_shape)

            self._add_gamma_weight(input_shape)
            self._add_beta_weight(input_shape)
            self.built = True
            super().build(input_shape)

        def call(self, inputs, training=None):
            # Training=none is just for compat with batchnorm signature call
            input_shape = tf.keras.backend.int_shape(inputs)
            tensor_input_shape = tf.shape(inputs)

            reshaped_inputs, group_shape = self._reshape_into_groups(
                inputs, input_shape, tensor_input_shape
            )

            normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

            outputs = tf.reshape(normalized_inputs, tensor_input_shape)

            return outputs

        def get_config(self):
            config = {
                "groups": self.groups,
                "axis": self.axis,
                "epsilon": self.epsilon,
                "center": self.center,
                "scale": self.scale,
                "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
                "gamma_initializer": tf.keras.initializers.serialize(
                    self.gamma_initializer
                ),
                "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
                "gamma_regularizer": tf.keras.regularizers.serialize(
                    self.gamma_regularizer
                ),
                "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
                "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
            }
            base_config = super().get_config()
            return {**base_config, **config}

        def compute_output_shape(self, input_shape):
            return input_shape

        def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

            group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape

        def _apply_normalization(self, reshaped_inputs, input_shape):

            group_shape = tf.keras.backend.int_shape(reshaped_inputs)
            group_reduction_axes = list(range(1, len(group_shape)))
            axis = -2 if self.axis == -1 else self.axis - 1
            group_reduction_axes.pop(axis)

            mean, variance = tf.nn.moments(
                reshaped_inputs, group_reduction_axes, keepdims=True
            )

            gamma, beta = self._get_reshaped_weights(input_shape)
            normalized_inputs = tf.nn.batch_normalization(
                reshaped_inputs,
                mean=mean,
                variance=variance,
                scale=gamma,
                offset=beta,
                variance_epsilon=self.epsilon,
            )
            return normalized_inputs

        def _get_reshaped_weights(self, input_shape):
            broadcast_shape = self._create_broadcast_shape(input_shape)
            gamma = None
            beta = None
            if self.scale:
                gamma = tf.reshape(self.gamma, broadcast_shape)

            if self.center:
                beta = tf.reshape(self.beta, broadcast_shape)
            return gamma, beta

        def _check_if_input_shape_is_none(self, input_shape):
            dim = input_shape[self.axis]
            if dim is None:
                raise ValueError(
                    "Axis " + str(self.axis) + " of "
                                               "input tensor should have a defined dimension "
                                               "but the layer received an input with shape " + str(input_shape) + "."
                )

        def _set_number_of_groups_for_instance_norm(self, input_shape):
            dim = input_shape[self.axis]

            if self.groups == -1:
                self.groups = dim

        def _check_size_of_dimensions(self, input_shape):

            dim = input_shape[self.axis]
            if dim < self.groups:
                raise ValueError(
                    "Number of groups (" + str(self.groups) + ") cannot be "
                                                              "more than the number of channels (" + str(dim) + ")."
                )

            if dim % self.groups != 0:
                raise ValueError(
                    "Number of groups (" + str(self.groups) + ") must be a "
                                                              "multiple of the number of channels (" + str(dim) + ")."
                )

        def _check_axis(self):

            if self.axis == 0:
                raise ValueError(
                    "You are trying to normalize your batch axis. Do you want to "
                    "use tf.layer.batch_normalization instead"
                )

        def _create_input_spec(self, input_shape):

            dim = input_shape[self.axis]
            self.input_spec = tf.keras.layers.InputSpec(
                ndim=len(input_shape), axes={self.axis: dim}
            )

        def _add_gamma_weight(self, input_shape):

            dim = input_shape[self.axis]
            shape = (dim,)

            if self.scale:
                self.gamma = self.add_weight(
                    shape=shape,
                    name="gamma",
                    initializer=self.gamma_initializer,
                    regularizer=self.gamma_regularizer,
                    constraint=self.gamma_constraint,
                )
            else:
                self.gamma = None

        def _add_beta_weight(self, input_shape):

            dim = input_shape[self.axis]
            shape = (dim,)

            if self.center:
                self.beta = self.add_weight(
                    shape=shape,
                    name="beta",
                    initializer=self.beta_initializer,
                    regularizer=self.beta_regularizer,
                    constraint=self.beta_constraint,
                )
            else:
                self.beta = None

        def _create_broadcast_shape(self, input_shape):
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
            return broadcast_shape

    class TransformBlock(tf.keras.Model):

        def __init__(self, features,
                     norm_type,
                     momentum=0.9,
                     virtual_batch_size=None,
                     groups=2,
                     block_name='',
                     **kwargs):
            super(TransformBlock, self).__init__(**kwargs)
            self.features = features
            self.norm_type = norm_type
            self.momentum = momentum
            self.groups = groups
            self.virtual_batch_size = virtual_batch_size
            self.transform = tf.keras.layers.Dense(self.features, use_bias=False,
                                                   name=f'transformblock_dense_{block_name}')
            if norm_type == 'batch':
                self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=momentum,
                                                             virtual_batch_size=virtual_batch_size,
                                                             name=f'transformblock_bn_{block_name}')
            else:
                self.bn = GroupNormalization(axis=-1, groups=self.groups, name=f'transformblock_gn_{block_name}')

        def call(self, inputs, training=None):
            x = self.transform(inputs)
            x = self.bn(x, training=training)
            return x

    class TabNet(tf.keras.Model):

        def __init__(self, feature_columns,
                     feature_dim=64,
                     output_dim=64,
                     num_features=None,
                     num_decision_steps=5,
                     relaxation_factor=1.5,
                     sparsity_coefficient=1e-5,
                     norm_type='group',
                     batch_momentum=0.98,
                     virtual_batch_size=None,
                     num_groups=2,
                     epsilon=1e-5,
                     **kwargs):
            super(TabNet, self).__init__(**kwargs)
            if feature_columns is not None:
                if type(feature_columns) not in (list, tuple): raise ValueError(
                    "`feature_columns` must be a list or a tuple.")
                if len(feature_columns) == 0: raise ValueError(
                    "`feature_columns` must be contain at least 1 tf.feature_column !")
                if num_features is None:
                    num_features = len(feature_columns)
                else:
                    num_features = int(num_features)
            else:
                if num_features is None: raise ValueError(
                    "If `feature_columns` is None, then `num_features` cannot be None.")
            if num_decision_steps < 1: raise ValueError("Num decision steps must be greater than 0.")
            if feature_dim < output_dim: raise ValueError(
                "To compute `features_for_coef`, feature_dim must be larger than output dim")
            feature_dim = int(feature_dim)
            output_dim = int(output_dim)
            num_decision_steps = int(num_decision_steps)
            relaxation_factor = float(relaxation_factor)
            sparsity_coefficient = float(sparsity_coefficient)
            batch_momentum = float(batch_momentum)
            num_groups = max(1, int(num_groups))
            epsilon = float(epsilon)
            if relaxation_factor < 0.: raise ValueError("`relaxation_factor` cannot be negative !")
            if sparsity_coefficient < 0.: raise ValueError("`sparsity_coefficient` cannot be negative !")
            if virtual_batch_size is not None: virtual_batch_size = int(virtual_batch_size)
            if norm_type not in ['batch', 'group']: raise ValueError("`norm_type` must be either `batch` or `group`")
            self.feature_columns = feature_columns
            self.num_features = num_features
            self.feature_dim = feature_dim
            self.output_dim = output_dim
            self.num_decision_steps = num_decision_steps
            self.relaxation_factor = relaxation_factor
            self.sparsity_coefficient = sparsity_coefficient
            self.norm_type = norm_type
            self.batch_momentum = batch_momentum
            self.virtual_batch_size = virtual_batch_size
            self.num_groups = num_groups
            self.epsilon = epsilon
            if self.feature_columns is not None:
                self.input_features = tf.keras.layers.DenseFeatures(feature_columns, trainable=True)
                if self.norm_type == 'batch':
                    self.input_bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=batch_momentum,
                                                                       name='input_bn')
                else:
                    self.input_bn = GroupNormalization(axis=-1, groups=self.num_groups, name='input_gn')
            else:
                self.input_features = None
                self.input_bn = None
            self.transform_f1 = TransformBlock(2 * self.feature_dim, self.norm_type, self.batch_momentum,
                                               self.virtual_batch_size, self.num_groups, block_name='f1')
            self.transform_f2 = TransformBlock(2 * self.feature_dim, self.norm_type, self.batch_momentum,
                                               self.virtual_batch_size, self.num_groups, block_name='f2')
            self.transform_f3_list = [
                TransformBlock(2 * self.feature_dim, self.norm_type, self.batch_momentum, self.virtual_batch_size,
                               self.num_groups, block_name=f'f3_{i}') for i in range(self.num_decision_steps)]
            self.transform_f4_list = [
                TransformBlock(2 * self.feature_dim, self.norm_type, self.batch_momentum, self.virtual_batch_size,
                               self.num_groups, block_name=f'f4_{i}') for i in range(self.num_decision_steps)]
            self.transform_coef_list = [
                TransformBlock(self.num_features, self.norm_type, self.batch_momentum, self.virtual_batch_size,
                               self.num_groups, block_name=f'coef_{i}') for i in range(self.num_decision_steps - 1)]
            self._step_feature_selection_masks = None
            self._step_aggregate_feature_selection_mask = None

        def call(self, inputs, training=None):
            if self.input_features is not None:
                features = self.input_features(inputs)
                features = self.input_bn(features, training=training)
            else:
                features = inputs
            batch_size = tf.shape(features)[0]
            self._step_feature_selection_masks = []
            self._step_aggregate_feature_selection_mask = None
            output_aggregated = tf.zeros([batch_size, self.output_dim])
            masked_features = features
            mask_values = tf.zeros([batch_size, self.num_features])
            aggregated_mask_values = tf.zeros([batch_size, self.num_features])
            complementary_aggregated_mask_values = tf.ones([batch_size, self.num_features])
            total_entropy = 0.0
            entropy_loss = 0.
            for ni in range(self.num_decision_steps):
                transform_f1 = self.transform_f1(masked_features, training=training)
                transform_f1 = glu(transform_f1, self.feature_dim)
                transform_f2 = self.transform_f2(transform_f1, training=training)
                transform_f2 = (glu(transform_f2, self.feature_dim) + transform_f1) * tf.math.sqrt(0.5)
                transform_f3 = self.transform_f3_list[ni](transform_f2, training=training)
                transform_f3 = (glu(transform_f3, self.feature_dim) + transform_f2) * tf.math.sqrt(0.5)
                transform_f4 = self.transform_f4_list[ni](transform_f3, training=training)
                transform_f4 = (glu(transform_f4, self.feature_dim) + transform_f3) * tf.math.sqrt(0.5)
                if (ni > 0 or self.num_decision_steps == 1):
                    decision_out = tf.nn.relu(transform_f4[:, :self.output_dim])
                    output_aggregated += decision_out
                    scale_agg = tf.reduce_sum(decision_out, axis=1, keepdims=True)
                    if self.num_decision_steps > 1: scale_agg = scale_agg / tf.cast(self.num_decision_steps - 1,
                                                                                    tf.float32)
                    aggregated_mask_values += mask_values * scale_agg
                features_for_coef = transform_f4[:, self.output_dim:]
                if ni < (self.num_decision_steps - 1):
                    mask_values = self.transform_coef_list[ni](features_for_coef, training=training)
                    mask_values *= complementary_aggregated_mask_values
                    mask_values = sparsemax(mask_values, axis=-1)
                    complementary_aggregated_mask_values *= (self.relaxation_factor - mask_values)
                    total_entropy += tf.reduce_mean(
                        tf.reduce_sum(-mask_values * tf.math.log(mask_values + self.epsilon), axis=1)) / (
                                         tf.cast(self.num_decision_steps - 1, tf.float32))
                    entropy_loss = total_entropy
                    masked_features = tf.multiply(mask_values, features)
                    mask_at_step_i = tf.expand_dims(tf.expand_dims(mask_values, 0), 3)
                    self._step_feature_selection_masks.append(mask_at_step_i)
                else:
                    entropy_loss = 0.
            self.add_loss(self.sparsity_coefficient * entropy_loss)
            agg_mask = tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3)
            self._step_aggregate_feature_selection_mask = agg_mask
            return output_aggregated

        @property
        def feature_selection_masks(self):
            return self._step_feature_selection_masks

        @property
        def aggregate_feature_selection_mask(self):
            return self._step_aggregate_feature_selection_mask

    class TabNetClassifier(tf.keras.Model):

        def __init__(self, feature_columns,
                     num_classes,
                     num_features=None,
                     feature_dim=64,
                     output_dim=64,
                     num_decision_steps=5,
                     relaxation_factor=1.5,
                     sparsity_coefficient=1e-5,
                     norm_type='group',
                     batch_momentum=0.98,
                     virtual_batch_size=None,
                     num_groups=1,
                     epsilon=1e-5,
                     multi_label=False,
                     **kwargs):
            super(TabNetClassifier, self).__init__(**kwargs)
            self.num_classes = num_classes
            self.tabnet = TabNet(feature_columns=feature_columns,
                                 num_features=num_features,
                                 feature_dim=feature_dim,
                                 output_dim=output_dim,
                                 num_decision_steps=num_decision_steps,
                                 relaxation_factor=relaxation_factor,
                                 sparsity_coefficient=sparsity_coefficient,
                                 norm_type=norm_type,
                                 batch_momentum=batch_momentum,
                                 virtual_batch_size=virtual_batch_size,
                                 num_groups=num_groups,
                                 epsilon=epsilon,
                                 **kwargs)
            if multi_label:
                self.clf = tf.keras.layers.Dense(num_classes, activation='sigmoid', use_bias=False, name='classifier')
            else:
                self.clf = tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False, name='classifier')

        def call(self, inputs, training=None):
            self.activations = self.tabnet(inputs, training=training)
            out = self.clf(self.activations)
            return out

        def summary(self, *super_args, **super_kwargs):
            super().summary(*super_args, **super_kwargs)
            self.tabnet.summary(*super_args, **super_kwargs)

    class TabNetRegressor(tf.keras.Model):

        def __init__(self, feature_columns,
                     num_regressors,
                     num_features=None,
                     feature_dim=64,
                     output_dim=64,
                     num_decision_steps=5,
                     relaxation_factor=1.5,
                     sparsity_coefficient=1e-5,
                     norm_type='group',
                     batch_momentum=0.98,
                     virtual_batch_size=None,
                     num_groups=1,
                     epsilon=1e-5,
                     **kwargs):
            super(TabNetRegressor, self).__init__(**kwargs)
            self.num_regressors = num_regressors
            self.tabnet = TabNet(feature_columns=feature_columns,
                                 num_features=num_features,
                                 feature_dim=feature_dim,
                                 output_dim=output_dim,
                                 num_decision_steps=num_decision_steps,
                                 relaxation_factor=relaxation_factor,
                                 sparsity_coefficient=sparsity_coefficient,
                                 norm_type=norm_type,
                                 batch_momentum=batch_momentum,
                                 virtual_batch_size=virtual_batch_size,
                                 num_groups=num_groups,
                                 epsilon=epsilon,
                                 **kwargs)
            self.regressor = tf.keras.layers.Dense(num_regressors, use_bias=False, name='regressor')

        def call(self, inputs, training=None):
            self.activations = self.tabnet(inputs, training=training)
            out = self.regressor(self.activations)
            return out

        def summary(self, *super_args, **super_kwargs):
            super().summary(*super_args, **super_kwargs)
            self.tabnet.summary(*super_args, **super_kwargs)

    TabNetClassification = TabNetClassifier
    TabNetRegression = TabNetRegressor

    class StackedTabNet(tf.keras.Model):

        def __init__(self, feature_columns,
                     num_layers=1,
                     feature_dim=64,
                     output_dim=64,
                     num_features=None,
                     num_decision_steps=5,
                     relaxation_factor=1.5,
                     sparsity_coefficient=1e-5,
                     norm_type='group',
                     batch_momentum=0.98,
                     virtual_batch_size=None,
                     num_groups=2,
                     epsilon=1e-5,
                     **kwargs):
            super(StackedTabNet, self).__init__(**kwargs)
            if num_layers < 1: raise ValueError("`num_layers` cannot be less than 1")
            if type(feature_dim) not in [list, tuple]: feature_dim = [feature_dim] * num_layers
            if type(output_dim) not in [list, tuple]: output_dim = [output_dim] * num_layers
            if len(feature_dim) != num_layers: raise ValueError("`feature_dim` must be a list of length `num_layers`")
            if len(output_dim) != num_layers: raise ValueError("`output_dim` must be a list of length `num_layers`")
            self.num_layers = num_layers
            layers = []
            layers.append(TabNet(feature_columns=feature_columns,
                                 num_features=num_features,
                                 feature_dim=feature_dim[0],
                                 output_dim=output_dim[0],
                                 num_decision_steps=num_decision_steps,
                                 relaxation_factor=relaxation_factor,
                                 sparsity_coefficient=sparsity_coefficient,
                                 norm_type=norm_type,
                                 batch_momentum=batch_momentum,
                                 virtual_batch_size=virtual_batch_size,
                                 num_groups=num_groups,
                                 epsilon=epsilon))

            for layer_idx in range(1, num_layers):
                layers.append(TabNet(feature_columns=None,
                                     num_features=output_dim[layer_idx - 1],
                                     feature_dim=feature_dim[layer_idx],
                                     output_dim=output_dim[layer_idx],
                                     num_decision_steps=num_decision_steps,
                                     relaxation_factor=relaxation_factor,
                                     sparsity_coefficient=sparsity_coefficient,
                                     norm_type=norm_type,
                                     batch_momentum=batch_momentum,
                                     virtual_batch_size=virtual_batch_size,
                                     num_groups=num_groups,
                                     epsilon=epsilon))
            self.tabnet_layers = layers

        def call(self, inputs, training=None):
            x = self.tabnet_layers[0](inputs, training=training)
            for layer_idx in range(1, self.num_layers): x = self.tabnet_layers[layer_idx](x, training=training)
            return x

        @property
        def tabnets(self):
            return self.tabnet_layers

        @property
        def feature_selection_masks(self):
            return [tabnet.feature_selection_masks for tabnet in self.tabnet_layers]

        @property
        def aggregate_feature_selection_mask(self):
            return [tabnet.aggregate_feature_selection_mask for tabnet in self.tabnet_layers]

    class StackedTabNetClassifier(tf.keras.Model):

        def __init__(self, feature_columns,
                     num_classes,
                     num_layers=1,
                     feature_dim=64,
                     output_dim=64,
                     num_features=None,
                     num_decision_steps=5,
                     relaxation_factor=1.5,
                     sparsity_coefficient=1e-5,
                     norm_type='group',
                     batch_momentum=0.98,
                     virtual_batch_size=None,
                     num_groups=2,
                     epsilon=1e-5,
                     multi_label=False,
                     **kwargs):
            super(StackedTabNetClassifier, self).__init__(**kwargs)
            self.num_classes = num_classes
            self.stacked_tabnet = StackedTabNet(feature_columns=feature_columns,
                                                num_layers=num_layers,
                                                feature_dim=feature_dim,
                                                output_dim=output_dim,
                                                num_features=num_features,
                                                num_decision_steps=num_decision_steps,
                                                relaxation_factor=relaxation_factor,
                                                sparsity_coefficient=sparsity_coefficient,
                                                norm_type=norm_type,
                                                batch_momentum=batch_momentum,
                                                virtual_batch_size=virtual_batch_size,
                                                num_groups=num_groups,
                                                epsilon=epsilon)
            if multi_label:
                self.clf = tf.keras.layers.Dense(num_classes, activation='sigmoid', use_bias=False)
            else:
                self.clf = tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False)

        def call(self, inputs, training=None):
            self.activations = self.stacked_tabnet(inputs[0], training=training)
            acts = tf.keras.layers.Concatenate()([self.activations, inputs[1]])
            out = self.clf(acts)
            return out

    class StackedTabNetRegressor(tf.keras.Model):

        def __init__(self, feature_columns,
                     num_regressors,
                     num_layers=1,
                     feature_dim=64,
                     output_dim=64,
                     num_features=None,
                     num_decision_steps=5,
                     relaxation_factor=1.5,
                     sparsity_coefficient=1e-5,
                     norm_type='group',
                     batch_momentum=0.98,
                     virtual_batch_size=None,
                     num_groups=2,
                     epsilon=1e-5,
                     **kwargs):
            super(StackedTabNetRegressor, self).__init__(**kwargs)
            self.num_regressors = num_regressors
            self.stacked_tabnet = StackedTabNet(feature_columns=feature_columns,
                                                num_layers=num_layers,
                                                feature_dim=feature_dim,
                                                output_dim=output_dim,
                                                num_features=num_features,
                                                num_decision_steps=num_decision_steps,
                                                relaxation_factor=relaxation_factor,
                                                sparsity_coefficient=sparsity_coefficient,
                                                norm_type=norm_type,
                                                batch_momentum=batch_momentum,
                                                virtual_batch_size=virtual_batch_size,
                                                num_groups=num_groups,
                                                epsilon=epsilon)
            self.regressor = tf.keras.layers.Dense(num_regressors, use_bias=False)

        def call(self, inputs, training=None):
            self.activations = self.tabnet(inputs, training=training)
            out = self.regressor(self.activations)
            return out


    num_layers = 10
    feature_dim = 570
    output_dim = 18
    num_decision_steps = 5
    relax = 1.925846253615446
    sparse = 0.27260674774203586

    model = StackedTabNetClassifier(feature_columns=None, num_classes=206, num_layers=num_layers,
                                    feature_dim=feature_dim, output_dim=output_dim, num_features=n_features,
                                    num_decision_steps=num_decision_steps, relaxation_factor=relax,
                                    sparsity_coefficient=sparse, batch_momentum=0.98,
                                    virtual_batch_size=None, norm_type='group',
                                    num_groups=-1, multi_label=True)
    model.compile(optimizer=tfa.optimizers.SWA(tf.optimizers.Adam(lr=0.001), start_averaging=9, average_period=6), loss=weighted_logloss(weights), metrics=logloss)

    input_1_1 = layers.Input(shape = (n_features,), name = 'Input1')
    input_2_1 = layers.Input(shape = (n_features_2,), name = 'Input2')
    mask_1 = layers.Dropout(0.3)((tf.ones_like(input_1_1)))
    mask_2 = layers.Dropout(0.3)((tf.ones_like(input_2_1)))

    encoded = model([input_1_1 * mask_1, input_2_1 * mask_2])

    head_4 = Sequential([
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(256, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(n_features, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        layers.Dense(n_features, activation="sigmoid")
        ],name='Head4')
    decoded = head_4(encoded)
    reconstruction_1 = decoded * (1 - mask_1) + input_1_1 * mask_1

    head_5 = Sequential([
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(256, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(n_features_2, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        layers.Dense(n_features_2, activation="sigmoid")
        ],name='Head5')
    decoded = head_5(encoded)
    reconstruction_2 = decoded * (1 - mask_2) + input_2_1 * mask_2

    loss = tf.reduce_mean(tf.math.abs(input_1_1 - reconstruction_1)) + tf.reduce_mean(tf.math.abs(input_2_1 - reconstruction_2))
    ae_model = Model(inputs=[input_1_1, input_2_1], outputs=[loss])
    ae_model.compile(optimizer=tf.optimizers.Adam(), loss=lambda t, y: y)

    return model, ae_model

def create_tabnet_3(n_features, n_features_2, n_labels, label_smoothing = 0.0005, brute_force=False, weights=None):
    def register_keras_custom_object(cls):
        tf.keras.utils.get_custom_objects()[cls.__name__] = cls
        return cls

    def glu(x, n_units=None):
        if n_units is None: n_units = tf.shape(x)[-1] // 2
        return x[..., :n_units] * tf.nn.sigmoid(x[..., n_units:])

    @register_keras_custom_object
    @tf.function
    def sparsemax(logits, axis):
        logits = tf.convert_to_tensor(logits, name="logits")
        shape = logits.get_shape()
        rank = shape.rank
        is_last_axis = (axis == -1) or (axis == rank - 1)
        if is_last_axis:
            output = _compute_2d_sparsemax(logits)
            output.set_shape(shape)
            return output
        rank_op = tf.rank(logits)
        axis_norm = axis % rank
        logits = _swap_axis(logits, axis_norm, tf.math.subtract(rank_op, 1))
        output = _compute_2d_sparsemax(logits)
        output = _swap_axis(output, axis_norm, tf.math.subtract(rank_op, 1))
        output.set_shape(shape)
        return output

    def _swap_axis(logits, dim_index, last_index, **kwargs):
        return tf.transpose(
            logits,
            tf.concat(
                [
                    tf.range(dim_index),
                    [last_index],
                    tf.range(dim_index + 1, last_index),
                    [dim_index],
                ],
                0,
            ),
            **kwargs,
        )

    def _compute_2d_sparsemax(logits):
        shape_op = tf.shape(logits)
        obs = tf.math.reduce_prod(shape_op[:-1])
        dims = shape_op[-1]
        z = tf.reshape(logits, [obs, dims])
        z_sorted, _ = tf.nn.top_k(z, k=dims)
        z_cumsum = tf.math.cumsum(z_sorted, axis=-1)
        k = tf.range(1, tf.cast(dims, logits.dtype) + 1, dtype=logits.dtype)
        z_check = 1 + k * z_sorted > z_cumsum
        k_z = tf.math.reduce_sum(tf.cast(z_check, tf.int32), axis=-1)
        k_z_safe = tf.math.maximum(k_z, 1)
        indices = tf.stack([tf.range(0, obs), tf.reshape(k_z_safe, [-1]) - 1], axis=1)
        tau_sum = tf.gather_nd(z_cumsum, indices)
        tau_z = (tau_sum - 1) / tf.cast(k_z, logits.dtype)
        p = tf.math.maximum(tf.cast(0, logits.dtype), z - tf.expand_dims(tau_z, -1))
        p_safe = tf.where(
            tf.expand_dims(
                tf.math.logical_or(tf.math.equal(k_z, 0), tf.math.is_nan(z_cumsum[:, -1])),
                axis=-1,
            ),
            tf.fill([obs, dims], tf.cast(float("nan"), logits.dtype)),
            p,
        )
        p_safe = tf.reshape(p_safe, shape_op)
        return p_safe

    class GhostBatchNormalization(tf.keras.Model):
        def __init__(
                self, virtual_divider: int = 1, momentum: float = 0.9, epsilon: float = 1e-5
        ):
            super(GhostBatchNormalization, self).__init__()
            self.virtual_divider = virtual_divider
            self.bn = BatchNormInferenceWeighting(momentum=momentum)

        def call(self, x, training: bool = None, alpha: float = 0.0):
            if training:
                chunks = tf.split(x, self.virtual_divider)
                x = [self.bn(x, training=True) for x in chunks]
                return tf.concat(x, 0)
            return self.bn(x, training=False, alpha=alpha)

        @property
        def moving_mean(self):
            return self.bn.moving_mean

        @property
        def moving_variance(self):
            return self.bn.moving_variance

    class BatchNormInferenceWeighting(tf.keras.layers.Layer):
        def __init__(self, momentum: float = 0.9, epsilon: float = None):
            super(BatchNormInferenceWeighting, self).__init__()
            self.momentum = momentum
            self.epsilon = tf.keras.backend.epsilon() if epsilon is None else epsilon

        def build(self, input_shape):
            channels = input_shape[-1]

            self.gamma = tf.Variable(
                initial_value=tf.ones((channels,), tf.float32), trainable=True,
            )
            self.beta = tf.Variable(
                initial_value=tf.zeros((channels,), tf.float32), trainable=True,
            )

            self.moving_mean = tf.Variable(
                initial_value=tf.zeros((channels,), tf.float32), trainable=False,
            )
            self.moving_mean_of_squares = tf.Variable(
                initial_value=tf.zeros((channels,), tf.float32), trainable=False,
            )

        def __update_moving(self, var, value):
            var.assign(var * self.momentum + (1 - self.momentum) * value)

        def __apply_normalization(self, x, mean, variance):
            return self.gamma * (x - mean) / tf.sqrt(variance + self.epsilon) + self.beta

        def call(self, x, training: bool = None, alpha: float = 0.0):
            mean = tf.reduce_mean(x, axis=0)
            mean_of_squares = tf.reduce_mean(tf.pow(x, 2), axis=0)

            if training:
                # update moving stats
                self.__update_moving(self.moving_mean, mean)
                self.__update_moving(self.moving_mean_of_squares, mean_of_squares)

                variance = mean_of_squares - tf.pow(mean, 2)
                x = self.__apply_normalization(x, mean, variance)
            else:
                mean = alpha * mean + (1 - alpha) * self.moving_mean
                variance = (
                                   alpha * mean_of_squares + (1 - alpha) * self.moving_mean_of_squares
                           ) - tf.pow(mean, 2)
                x = self.__apply_normalization(x, mean, variance)

            return x

    class FeatureBlock(tf.keras.Model):
        def __init__(
                self,
                feature_dim: int,
                apply_glu: bool = True,
                bn_momentum: float = 0.9,
                bn_virtual_divider: int = 32,
                fc: tf.keras.layers.Layer = None,
                epsilon: float = 1e-5,
        ):
            super(FeatureBlock, self).__init__()
            self.apply_gpu = apply_glu
            self.feature_dim = feature_dim
            units = feature_dim * 2 if apply_glu else feature_dim

            self.fc = tf.keras.layers.Dense(units, use_bias=False) if fc is None else fc
            self.bn = GhostBatchNormalization(
                virtual_divider=bn_virtual_divider, momentum=bn_momentum
            )

        def call(self, x, training: bool = None, alpha: float = 0.0):
            x = self.fc(x)
            x = self.bn(x, training=training, alpha=alpha)
            if self.apply_gpu:
                return glu(x, self.feature_dim)
            return x

    class AttentiveTransformer(tf.keras.Model):
        def __init__(self, feature_dim: int, bn_momentum: float, bn_virtual_divider: int):
            super(AttentiveTransformer, self).__init__()
            self.block = FeatureBlock(
                feature_dim,
                bn_momentum=bn_momentum,
                bn_virtual_divider=bn_virtual_divider,
                apply_glu=False,
            )

        def call(self, x, prior_scales, training=None, alpha: float = 0.0):
            x = self.block(x, training=training, alpha=alpha)
            return sparsemax(x * prior_scales, -1)

    class FeatureTransformer(tf.keras.Model):
        def __init__(
                self,
                feature_dim: int,
                fcs: List[tf.keras.layers.Layer] = [],
                n_total: int = 4,
                n_shared: int = 2,
                bn_momentum: float = 0.9,
                bn_virtual_divider: int = 1,
        ):
            super(FeatureTransformer, self).__init__()
            self.n_total, self.n_shared = n_total, n_shared

            kargs = {
                "feature_dim": feature_dim,
                "bn_momentum": bn_momentum,
                "bn_virtual_divider": bn_virtual_divider,
            }

            # build blocks
            self.blocks: List[FeatureBlock] = []
            for n in range(n_total):
                # some shared blocks
                if fcs and n < len(fcs):
                    self.blocks.append(FeatureBlock(**kargs, fc=fcs[n]))
                # build new blocks
                else:
                    self.blocks.append(FeatureBlock(**kargs))

        def call(
                self, x: tf.Tensor, training: bool = None, alpha: float = 0.0
        ) -> tf.Tensor:
            x = self.blocks[0](x, training=training, alpha=alpha)
            for n in range(1, self.n_total):
                x = x * tf.sqrt(0.5) + self.blocks[n](x, training=training, alpha=alpha)
            return x

        @property
        def shared_fcs(self):
            return [self.blocks[i].fc for i in range(self.n_shared)]

    @register_keras_custom_object
    class GroupNormalization(tf.keras.layers.Layer):

        def __init__(
                self,
                groups: int = 2,
                axis: int = -1,
                epsilon: float = 1e-3,
                center: bool = True,
                scale: bool = True,
                beta_initializer="zeros",
                gamma_initializer="ones",
                beta_regularizer=None,
                gamma_regularizer=None,
                beta_constraint=None,
                gamma_constraint=None,
                **kwargs
        ):
            super().__init__(**kwargs)
            self.supports_masking = True
            self.groups = groups
            self.axis = axis
            self.epsilon = epsilon
            self.center = center
            self.scale = scale
            self.beta_initializer = tf.keras.initializers.get(beta_initializer)
            self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
            self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
            self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
            self.beta_constraint = tf.keras.constraints.get(beta_constraint)
            self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
            self._check_axis()

        def build(self, input_shape):

            self._check_if_input_shape_is_none(input_shape)
            self._set_number_of_groups_for_instance_norm(input_shape)
            self._check_size_of_dimensions(input_shape)
            self._create_input_spec(input_shape)

            self._add_gamma_weight(input_shape)
            self._add_beta_weight(input_shape)
            self.built = True
            super().build(input_shape)

        def call(self, inputs, training=None):
            # Training=none is just for compat with batchnorm signature call
            input_shape = tf.keras.backend.int_shape(inputs)
            tensor_input_shape = tf.shape(inputs)

            reshaped_inputs, group_shape = self._reshape_into_groups(
                inputs, input_shape, tensor_input_shape
            )

            normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)

            outputs = tf.reshape(normalized_inputs, tensor_input_shape)

            return outputs

        def get_config(self):
            config = {
                "groups": self.groups,
                "axis": self.axis,
                "epsilon": self.epsilon,
                "center": self.center,
                "scale": self.scale,
                "beta_initializer": tf.keras.initializers.serialize(self.beta_initializer),
                "gamma_initializer": tf.keras.initializers.serialize(
                    self.gamma_initializer
                ),
                "beta_regularizer": tf.keras.regularizers.serialize(self.beta_regularizer),
                "gamma_regularizer": tf.keras.regularizers.serialize(
                    self.gamma_regularizer
                ),
                "beta_constraint": tf.keras.constraints.serialize(self.beta_constraint),
                "gamma_constraint": tf.keras.constraints.serialize(self.gamma_constraint),
            }
            base_config = super().get_config()
            return {**base_config, **config}

        def compute_output_shape(self, input_shape):
            return input_shape

        def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

            group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
            group_shape[self.axis] = input_shape[self.axis] // self.groups
            group_shape.insert(self.axis, self.groups)
            group_shape = tf.stack(group_shape)
            reshaped_inputs = tf.reshape(inputs, group_shape)
            return reshaped_inputs, group_shape

        def _apply_normalization(self, reshaped_inputs, input_shape):

            group_shape = tf.keras.backend.int_shape(reshaped_inputs)
            group_reduction_axes = list(range(1, len(group_shape)))
            axis = -2 if self.axis == -1 else self.axis - 1
            group_reduction_axes.pop(axis)

            mean, variance = tf.nn.moments(
                reshaped_inputs, group_reduction_axes, keepdims=True
            )

            gamma, beta = self._get_reshaped_weights(input_shape)
            normalized_inputs = tf.nn.batch_normalization(
                reshaped_inputs,
                mean=mean,
                variance=variance,
                scale=gamma,
                offset=beta,
                variance_epsilon=self.epsilon,
            )
            return normalized_inputs

        def _get_reshaped_weights(self, input_shape):
            broadcast_shape = self._create_broadcast_shape(input_shape)
            gamma = None
            beta = None
            if self.scale:
                gamma = tf.reshape(self.gamma, broadcast_shape)

            if self.center:
                beta = tf.reshape(self.beta, broadcast_shape)
            return gamma, beta

        def _check_if_input_shape_is_none(self, input_shape):
            dim = input_shape[self.axis]
            if dim is None:
                raise ValueError(
                    "Axis " + str(self.axis) + " of "
                                               "input tensor should have a defined dimension "
                                               "but the layer received an input with shape " + str(input_shape) + "."
                )

        def _set_number_of_groups_for_instance_norm(self, input_shape):
            dim = input_shape[self.axis]

            if self.groups == -1:
                self.groups = dim

        def _check_size_of_dimensions(self, input_shape):

            dim = input_shape[self.axis]
            if dim < self.groups:
                raise ValueError(
                    "Number of groups (" + str(self.groups) + ") cannot be "
                                                              "more than the number of channels (" + str(dim) + ")."
                )

            if dim % self.groups != 0:
                raise ValueError(
                    "Number of groups (" + str(self.groups) + ") must be a "
                                                              "multiple of the number of channels (" + str(dim) + ")."
                )

        def _check_axis(self):

            if self.axis == 0:
                raise ValueError(
                    "You are trying to normalize your batch axis. Do you want to "
                    "use tf.layer.batch_normalization instead"
                )

        def _create_input_spec(self, input_shape):

            dim = input_shape[self.axis]
            self.input_spec = tf.keras.layers.InputSpec(
                ndim=len(input_shape), axes={self.axis: dim}
            )

        def _add_gamma_weight(self, input_shape):

            dim = input_shape[self.axis]
            shape = (dim,)

            if self.scale:
                self.gamma = self.add_weight(
                    shape=shape,
                    name="gamma",
                    initializer=self.gamma_initializer,
                    regularizer=self.gamma_regularizer,
                    constraint=self.gamma_constraint,
                )
            else:
                self.gamma = None

        def _add_beta_weight(self, input_shape):

            dim = input_shape[self.axis]
            shape = (dim,)

            if self.center:
                self.beta = self.add_weight(
                    shape=shape,
                    name="beta",
                    initializer=self.beta_initializer,
                    regularizer=self.beta_regularizer,
                    constraint=self.beta_constraint,
                )
            else:
                self.beta = None

        def _create_broadcast_shape(self, input_shape):
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
            broadcast_shape.insert(self.axis, self.groups)
            return broadcast_shape

    class TransformBlock(tf.keras.Model):

        def __init__(self, features,
                     norm_type,
                     momentum=0.9,
                     virtual_batch_size=None,
                     groups=2,
                     block_name='',
                     **kwargs):
            super(TransformBlock, self).__init__(**kwargs)
            self.features = features
            self.norm_type = norm_type
            self.momentum = momentum
            self.groups = groups
            self.virtual_batch_size = virtual_batch_size
            self.transform = tf.keras.layers.Dense(self.features, use_bias=False,
                                                   name=f'transformblock_dense_{block_name}')
            if norm_type == 'batch':
                self.bn = tf.keras.layers.BatchNormalization(axis=-1, momentum=momentum,
                                                             virtual_batch_size=virtual_batch_size,
                                                             name=f'transformblock_bn_{block_name}')
            else:
                self.bn = GroupNormalization(axis=-1, groups=self.groups, name=f'transformblock_gn_{block_name}')

        def call(self, inputs, training=None):
            x = self.transform(inputs)
            x = self.bn(x, training=training)
            return x

    class TabNet(tf.keras.Model):
        def __init__(
                self,
                num_features: int,
                feature_dim: int,
                output_dim: int,
                feature_columns: List = None,
                num_decision_steps: int = 1,
                n_total: int = 4,
                n_shared: int = 2,
                relaxation_factor: float = 1.5,
                bn_epsilon: float = 1e-5,
                batch_momentum: float = 0.7,
                bn_virtual_divider: int = 1,
                sparsity_coefficient=0,
        ):
            super(TabNet, self).__init__()
            self.output_dim, self.num_features = output_dim, num_features
            self.n_step, self.relaxation_factor = num_decision_steps, relaxation_factor
            self.feature_columns = feature_columns
            if feature_columns is not None:
                self.input_features = tf.keras.layers.DenseFeatures(feature_columns)
            self.bn = tf.keras.layers.BatchNormalization(
                momentum=batch_momentum, epsilon=bn_epsilon
            )
            kargs = {
                "feature_dim": feature_dim + output_dim,
                "n_total": n_total,
                "n_shared": n_shared,
                "bn_momentum": batch_momentum,
                "bn_virtual_divider": bn_virtual_divider,
            }
            self.feature_transforms = [FeatureTransformer(**kargs)]
            self.attentive_transforms = []
            for i in range(num_decision_steps):
                self.feature_transforms.append(FeatureTransformer(**kargs, fcs=self.feature_transforms[0].shared_fcs))
                self.attentive_transforms.append(AttentiveTransformer(num_features, batch_momentum, bn_virtual_divider))

        def call(self, features, training=None, alpha=0.0):
            if self.feature_columns is not None:
                features = self.input_features(features)
            bs = tf.shape(features)[0]
            out_agg = tf.zeros((bs, self.output_dim))
            prior_scales = tf.ones((bs, self.num_features))
            masks = []
            features = self.bn(features, training=training)
            masked_features = features
            total_entropy = 0.0
            for step_i in range(self.n_step + 1):
                x = self.feature_transforms[step_i](
                    masked_features, training=training, alpha=alpha
                )
                if step_i > 0:
                    out = tf.keras.activations.relu(x[:, : self.output_dim])
                    out_agg += out
                if step_i < self.n_step:
                    x_for_mask = x[:, self.output_dim:]
                    mask_values = self.attentive_transforms[step_i](
                        x_for_mask, prior_scales, training=training, alpha=alpha
                    )
                    prior_scales *= self.relaxation_factor - mask_values
                    masked_features = tf.multiply(mask_values, features)
                    total_entropy = tf.reduce_mean(
                        tf.reduce_sum(
                            tf.multiply(mask_values, tf.math.log(mask_values + 1e-15)),
                            axis=1,
                        )
                    )
                    masks.append(tf.expand_dims(tf.expand_dims(mask_values, 0), 3))
            loss = total_entropy / self.n_step
            self.add_loss(loss)
            return out_agg  # , loss, masks

    class TabNetClassifier(tf.keras.Model):
        def __init__(
                self,
                num_features: int,
                feature_dim: int,
                output_dim: int,
                n_classes: int,
                feature_columns: List = None,
                n_step: int = 1,
                n_total: int = 4,
                n_shared: int = 2,
                relaxation_factor: float = 1.5,
                sparsity_coefficient: float = 1e-5,
                bn_epsilon: float = 1e-5,
                bn_momentum: float = 0.7,
                bn_virtual_divider: int = 32,
                dp: float = None,
                **kwargs
        ):
            super(TabNetClassifier, self).__init__()

            self.configs = {
                "num_features": num_features,
                "feature_dim": feature_dim,
                "output_dim": output_dim,
                "n_classes": n_classes,
                "feature_columns": feature_columns,
                "n_step": n_step,
                "n_total": n_total,
                "n_shared": n_shared,
                "relaxation_factor": relaxation_factor,
                "sparsity_coefficient": sparsity_coefficient,
                "bn_epsilon": bn_epsilon,
                "bn_momentum": bn_momentum,
                "bn_virtual_divider": bn_virtual_divider,
                "dp": dp,
            }
            for k, v in kwargs.items():
                self.configs[k] = v

            self.sparsity_coefficient = sparsity_coefficient

            self.model = TabNet(
                feature_columns=feature_columns,
                num_features=num_features,
                feature_dim=feature_dim,
                output_dim=output_dim,
                num_decision_steps=n_step,
                relaxation_factor=relaxation_factor,
                bn_epsilon=bn_epsilon,
                batch_momentum=bn_momentum,
                bn_virtual_divider=bn_virtual_divider,
            )
            self.dp = tf.keras.layers.Dropout(dp) if dp is not None else dp
            self.head = tf.keras.layers.Dense(n_classes, activation=None, use_bias=False)

        def call(self, x, training: bool = None, alpha: float = 0.0):
            out, sparse_loss, _ = self.model(x, training=training, alpha=alpha)
            if self.dp is not None:
                out = self.dp(out, training=training)
            y = self.head(out, training=training)

            if training:
                self.add_loss(-self.sparsity_coefficient * sparse_loss)

            return y

        def get_config(self):
            return self.configs

        def save_to_directory(self, path_to_folder):
            self.save_weights(os.path.join(path_to_folder, "ckpt"), overwrite=True)
            with open(os.path.join(path_to_folder, "configs.pickle"), "wb") as f:
                pickle.dump(self.configs, f)

        @classmethod
        def load_from_directory(cls, path_to_folder):
            with open(os.path.join(path_to_folder, "configs.pickle"), "rb") as f:
                configs = pickle.load(f)
            model: tf.keras.Model = cls(**configs)
            model.build((None, configs["num_features"]))
            load_status = model.load_weights(os.path.join(path_to_folder, "ckpt"))
            load_status.expect_partial()
            return model

    class TabNetRegressor(tf.keras.Model):

        def __init__(self, feature_columns,
                     num_regressors,
                     num_features=None,
                     feature_dim=64,
                     output_dim=64,
                     num_decision_steps=5,
                     relaxation_factor=1.5,
                     sparsity_coefficient=1e-5,
                     norm_type='group',
                     batch_momentum=0.98,
                     virtual_batch_size=None,
                     num_groups=1,
                     epsilon=1e-5,
                     **kwargs):
            super(TabNetRegressor, self).__init__(**kwargs)
            self.num_regressors = num_regressors
            self.tabnet = TabNet(feature_columns=feature_columns,
                                 num_features=num_features,
                                 feature_dim=feature_dim,
                                 output_dim=output_dim,
                                 num_decision_steps=num_decision_steps,
                                 relaxation_factor=relaxation_factor,
                                 sparsity_coefficient=sparsity_coefficient,
                                 norm_type=norm_type,
                                 batch_momentum=batch_momentum,
                                 virtual_batch_size=virtual_batch_size,
                                 num_groups=num_groups,
                                 epsilon=epsilon,
                                 **kwargs)
            self.regressor = tf.keras.layers.Dense(num_regressors, use_bias=False, name='regressor')

        def call(self, inputs, training=None):
            self.activations = self.tabnet(inputs, training=training)
            out = self.regressor(self.activations)
            return out

        def summary(self, *super_args, **super_kwargs):
            super().summary(*super_args, **super_kwargs)
            self.tabnet.summary(*super_args, **super_kwargs)

    TabNetClassification = TabNetClassifier
    TabNetRegression = TabNetRegressor

    class StackedTabNet(tf.keras.Model):

        def __init__(self, feature_columns,
                     num_layers=1,
                     feature_dim=64,
                     output_dim=64,
                     num_features=None,
                     num_decision_steps=5,
                     relaxation_factor=1.5,
                     sparsity_coefficient=1e-5,
                     norm_type='group',
                     batch_momentum=0.98,
                     virtual_batch_size=None,
                     num_groups=2,
                     epsilon=1e-5,
                     **kwargs):
            super(StackedTabNet, self).__init__(**kwargs)
            if num_layers < 1: raise ValueError("`num_layers` cannot be less than 1")
            if type(feature_dim) not in [list, tuple]: feature_dim = [feature_dim] * num_layers
            if type(output_dim) not in [list, tuple]: output_dim = [output_dim] * num_layers
            if len(feature_dim) != num_layers: raise ValueError("`feature_dim` must be a list of length `num_layers`")
            if len(output_dim) != num_layers: raise ValueError("`output_dim` must be a list of length `num_layers`")
            self.num_layers = num_layers
            layers = []
            layers.append(TabNet(feature_columns=feature_columns,
                                 num_features=num_features,
                                 feature_dim=feature_dim[0],
                                 output_dim=output_dim[0],
                                 num_decision_steps=num_decision_steps,
                                 relaxation_factor=relaxation_factor,
                                 sparsity_coefficient=sparsity_coefficient,
                                 # norm_type=norm_type,
                                 batch_momentum=batch_momentum))
            # virtual_batch_size=virtual_batch_size,
            # num_groups=num_groups,
            # epsilon=epsilon))

            for layer_idx in range(1, num_layers):
                layers.append(TabNet(feature_columns=None,
                                     num_features=output_dim[layer_idx - 1],
                                     feature_dim=feature_dim[layer_idx],
                                     output_dim=output_dim[layer_idx],
                                     num_decision_steps=num_decision_steps,
                                     relaxation_factor=relaxation_factor,
                                     sparsity_coefficient=sparsity_coefficient,
                                     # norm_type=norm_type,
                                     batch_momentum=batch_momentum))
                # virtual_batch_size=virtual_batch_size,
                # num_groups=num_groups,
                # epsilon=epsilon))
            self.tabnet_layers = layers

        def call(self, inputs, training=None):
            x = self.tabnet_layers[0](inputs, training=training)
            for layer_idx in range(1, self.num_layers): x = self.tabnet_layers[layer_idx](x, training=training)
            return x

        @property
        def tabnets(self):
            return self.tabnet_layers

        @property
        def feature_selection_masks(self):
            return [tabnet.feature_selection_masks for tabnet in self.tabnet_layers]

        @property
        def aggregate_feature_selection_mask(self):
            return [tabnet.aggregate_feature_selection_mask for tabnet in self.tabnet_layers]

    class StackedTabNetClassifier(tf.keras.Model):

        def __init__(self, feature_columns,
                     num_classes,
                     num_layers=1,
                     feature_dim=64,
                     output_dim=64,
                     num_features=None,
                     num_decision_steps=5,
                     relaxation_factor=1.5,
                     sparsity_coefficient=1e-5,
                     norm_type='group',
                     batch_momentum=0.98,
                     virtual_batch_size=None,
                     num_groups=2,
                     epsilon=1e-5,
                     multi_label=False,
                     **kwargs):
            super(StackedTabNetClassifier, self).__init__(**kwargs)
            self.num_classes = num_classes
            self.stacked_tabnet = StackedTabNet(feature_columns=feature_columns,
                                                num_layers=num_layers,
                                                feature_dim=feature_dim,
                                                output_dim=output_dim,
                                                num_features=num_features,
                                                num_decision_steps=num_decision_steps,
                                                relaxation_factor=relaxation_factor,
                                                sparsity_coefficient=sparsity_coefficient,
                                                norm_type=norm_type,
                                                batch_momentum=batch_momentum,
                                                virtual_batch_size=virtual_batch_size,
                                                num_groups=num_groups,
                                                epsilon=epsilon)
            if multi_label:
                self.clf = tf.keras.layers.Dense(num_classes, activation='sigmoid', use_bias=False)
            else:
                self.clf = tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=False)

        def call(self, inputs, training=None):
            self.activations = self.stacked_tabnet(inputs[0], training=training)
            acts = tf.keras.layers.Concatenate()([self.activations, inputs[1]])
            out = self.clf(acts)
            return out

    class StackedTabNetRegressor(tf.keras.Model):

        def __init__(self, feature_columns,
                     num_regressors,
                     num_layers=1,
                     feature_dim=64,
                     output_dim=64,
                     num_features=None,
                     num_decision_steps=5,
                     relaxation_factor=1.5,
                     sparsity_coefficient=1e-5,
                     norm_type='group',
                     batch_momentum=0.98,
                     virtual_batch_size=None,
                     num_groups=2,
                     epsilon=1e-5,
                     **kwargs):
            super(StackedTabNetRegressor, self).__init__(**kwargs)
            self.num_regressors = num_regressors
            self.stacked_tabnet = StackedTabNet(feature_columns=feature_columns,
                                                num_layers=num_layers,
                                                feature_dim=feature_dim,
                                                output_dim=output_dim,
                                                num_features=num_features,
                                                num_decision_steps=num_decision_steps,
                                                relaxation_factor=relaxation_factor,
                                                sparsity_coefficient=sparsity_coefficient,
                                                norm_type=norm_type,
                                                batch_momentum=batch_momentum,
                                                virtual_batch_size=virtual_batch_size,
                                                num_groups=num_groups,
                                                epsilon=epsilon)
            self.regressor = tf.keras.layers.Dense(num_regressors, use_bias=False)

        def call(self, inputs, training=None):
            self.activations = self.tabnet(inputs, training=training)
            out = self.regressor(self.activations)
            return out


    num_layers = 1
    feature_dim = 64
    output_dim = 64
    num_decision_steps = 5
    relax = 1.5
    sparse = 1e-5

    model = StackedTabNetClassifier(feature_columns=None, num_classes=206, num_layers=num_layers,
                                    feature_dim=feature_dim, output_dim=output_dim, num_features=n_features,
                                    num_decision_steps=num_decision_steps, relaxation_factor=relax,
                                    sparsity_coefficient=sparse, batch_momentum=0.98,
                                    virtual_batch_size=None, norm_type='group',
                                    num_groups=-1, multi_label=True)
    model.compile(optimizer='adam', loss=losses.BinaryCrossentropy(label_smoothing=label_smoothing), metrics=logloss)
    model.compile(optimizer=tfa.optimizers.SWA(tf.optimizers.Adam(lr=0.001), start_averaging=9, average_period=6), loss=weighted_logloss(weights), metrics=logloss)

    input_1_1 = layers.Input(shape = (n_features,), name = 'Input1')
    input_2_1 = layers.Input(shape = (n_features_2,), name = 'Input2')
    mask_1 = layers.Dropout(0.3)((tf.ones_like(input_1_1)))
    mask_2 = layers.Dropout(0.3)((tf.ones_like(input_2_1)))

    encoded = model([input_1_1 * mask_1, input_2_1 * mask_2])

    head_4 = Sequential([
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(256, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(n_features, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        layers.Dense(n_features, activation="sigmoid")
        ],name='Head4')
    decoded = head_4(encoded)
    reconstruction_1 = decoded * (1 - mask_1) + input_1_1 * mask_1

    head_5 = Sequential([
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(256, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(n_features_2, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        layers.Dense(n_features_2, activation="sigmoid")
        ],name='Head5')
    decoded = head_5(encoded)
    reconstruction_2 = decoded * (1 - mask_2) + input_2_1 * mask_2

    loss = tf.reduce_mean(tf.math.abs(input_1_1 - reconstruction_1)) + tf.reduce_mean(tf.math.abs(input_2_1 - reconstruction_2))
    ae_model = Model(inputs=[input_1_1, input_2_1], outputs=[loss])
    ae_model.compile(optimizer=tf.optimizers.Adam(), loss=lambda t, y: y)

    return model, ae_model

def create_tabnet_4(n_features, n_features_2, n_labels, label_smoothing = 0.0005, brute_force=False, weights=None):
    class GLUBlock(tf.keras.layers.Layer):
        def __init__(
                self,
                units=None,
                virtual_batch_size=128,
                momentum=0.02,
        ):
            super(GLUBlock, self).__init__()
            self.units = units
            self.virtual_batch_size = virtual_batch_size
            self.momentum = momentum

        def build(self, input_shape):
            if self.units is None:
                self.units = input_shape[-1]

            self.fc_outout = tf.keras.layers.Dense(self.units, use_bias=False)
            self.bn_outout = tf.keras.layers.BatchNormalization(
                virtual_batch_size=self.virtual_batch_size, momentum=self.momentum
            )

            self.fc_gate = tf.keras.layers.Dense(self.units, use_bias=False)
            self.bn_gate = tf.keras.layers.BatchNormalization(
                virtual_batch_size=self.virtual_batch_size, momentum=self.momentum
            )

        def call(self, inputs, training=None):
            output = self.bn_outout(self.fc_outout(inputs), training=training)
            gate = self.bn_gate(self.fc_gate(inputs), training=training)
            return output * tf.keras.activations.sigmoid(gate)  # GLU
    class FeatureTransformerBlock(tf.keras.layers.Layer):
        def __init__(
                self,
                units: Optional[int] = None,
                virtual_batch_size: Optional[int] = 128,
                momentum: Optional[float] = 0.02,
                skip: bool = False
        ):
            super(FeatureTransformerBlock, self).__init__()
            self.units = units
            self.virtual_batch_size = virtual_batch_size
            self.momentum = momentum
            self.skip = skip

        def build(self, input_shape: tf.TensorShape):
            if self.units is None:
                self.units = input_shape[-1]

            self.initial = GLUBlock(
                units=self.units,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
            )
            self.residual = GLUBlock(
                units=self.units,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
            )

        def call(
                self, inputs, training=None
        ):
            initial = self.initial(inputs, training=training)

            if self.skip:
                initial = (initial + inputs) * np.sqrt(0.5)

            residual = self.residual(initial, training=training)  # skip

            return (initial + residual) * np.sqrt(0.5)
    class AttentiveTransformer(tf.keras.layers.Layer):
        def __init__(
                self,
                units: Optional[int] = None,
                virtual_batch_size: Optional[int] = 128,
                momentum: Optional[float] = 0.02,
        ):
            super(AttentiveTransformer, self).__init__()
            self.units = units
            self.virtual_batch_size = virtual_batch_size
            self.momentum = momentum

        def build(self, input_shape):
            if self.units is None:
                self.units = input_shape[-1]

            self.fc = tf.keras.layers.Dense(self.units, use_bias=False)
            self.bn = tf.keras.layers.BatchNormalization(
                virtual_batch_size=self.virtual_batch_size, momentum=self.momentum
            )

        def call(
                self,
                inputs,
                priors,
                training=None):
            feature = self.bn(self.fc(inputs), training=training)
            output = feature * priors

            return tfa.activations.sparsemax(output)
    class TabNetStep(tf.keras.layers.Layer):
        def __init__(
                self,
                units=None,
                virtual_batch_size=128,
                momentum=0.02,
        ):
            super(TabNetStep, self).__init__()
            self.units = units
            self.virtual_batch_size = virtual_batch_size
            self.momentum = momentum

        def build(self, input_shape):
            if self.units is None:
                self.units = input_shape[-1]

            self.unique = FeatureTransformerBlock(
                units=self.units,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
                skip=True
            )
            self.attention = AttentiveTransformer(
                units=input_shape[-1],
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
            )

        def call(self, inputs, shared, priors, training=None):
            split = self.unique(shared, training=training)
            keys = self.attention(split, priors, training=training)
            masked = keys * inputs
            return split, masked, keys
    class TabNetEncoder(tf.keras.layers.Layer):
        def __init__(
                self,
                units: int = 1,
                n_steps: int = 3,
                n_features: int = 8,
                outputs: int = 1,
                gamma: float = 1.3,
                epsilon: float = 1e-8,
                sparsity: float = 1e-5,
                virtual_batch_size: Optional[int] = 128,
                momentum: Optional[float] = 0.02,
        ):
            super(TabNetEncoder, self).__init__()

            self.units = units
            self.n_steps = n_steps
            self.n_features = n_features
            self.virtual_batch_size = virtual_batch_size
            self.gamma = gamma
            self.epsilon = epsilon
            self.momentum = momentum
            self.sparsity = sparsity

        def build(self, input_shape):
            self.bn = tf.keras.layers.BatchNormalization(
                virtual_batch_size=self.virtual_batch_size, momentum=self.momentum
            )
            self.shared_block = FeatureTransformerBlock(
                units=self.n_features,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
            )
            self.initial_step = TabNetStep(
                units=self.n_features,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
            )
            self.steps = [
                TabNetStep(
                    units=self.n_features,
                    virtual_batch_size=self.virtual_batch_size,
                    momentum=self.momentum,
                )
                for _ in range(self.n_steps)
            ]
            self.final = tf.keras.layers.Dense(units=self.units, use_bias=False)

        def call(self, X, training=None):
            entropy_loss = 0.0
            encoded = 0.0
            output = 0.0
            importance = 0.0
            prior = tf.reduce_mean(tf.ones_like(X), axis=0)
            B = prior * self.bn(X, training=training)
            shared = self.shared_block(B, training=training)
            _, masked, keys = self.initial_step(B, shared, prior, training=training)
            for step in self.steps:
                entropy_loss += tf.reduce_mean(
                    tf.reduce_sum(-keys * tf.math.log(keys + self.epsilon), axis=-1)
                ) / tf.cast(self.n_steps, tf.float32)
                prior *= self.gamma - tf.reduce_mean(keys, axis=0)
                importance += keys
                shared = self.shared_block(masked, training=training)
                split, masked, keys = step(B, shared, prior, training=training)
                features = tf.keras.activations.relu(split)
                output += features
                encoded += split
            self.add_loss(self.sparsity * entropy_loss)
            prediction = self.final(output)
            return prediction, encoded, importance
    class TabNetDecoder(tf.keras.layers.Layer):
        def __init__(
                self,
                units=1,
                n_steps=3,
                n_features=8,
                outputs=1,
                gamma=1.3,
                epsilon=1e-8,
                sparsity=1e-5,
                virtual_batch_size=128,
                momentum=0.02):
            super(TabNetDecoder, self).__init__()
            self.units = units
            self.n_steps = n_steps
            self.n_features = n_features
            self.virtual_batch_size = virtual_batch_size
            self.momentum = momentum

        def build(self, input_shape: tf.TensorShape):
            self.shared_block = FeatureTransformerBlock(
                units=self.n_features,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
            )
            self.steps = [
                FeatureTransformerBlock(
                    units=self.n_features,
                    virtual_batch_size=self.virtual_batch_size,
                    momentum=self.momentum,
                )
                for _ in range(self.n_steps)
            ]
            self.fc = [tf.keras.layers.Dense(units=self.units) for _ in range(self.n_steps)]

        def call(self, X, training=None):
            decoded = 0.0

            for ftb, fc in zip(self.steps, self.fc):
                shared = self.shared_block(X, training=training)
                feature = ftb(shared, training=training)
                output = fc(feature)

                decoded += output
            return decoded
    @tf.function
    def identity(x):
        return x
    class TabNetModel(tf.keras.Model):
        def __init__(
                self,
                outputs: int = 1,
                n_steps: int = 3,
                n_features: int = 8,
                gamma: float = 1.3,
                epsilon: float = 1e-8,
                sparsity: float = 1e-5,
                feature_column: Optional[tf.keras.layers.DenseFeatures] = None,
                pretrained_encoder: Optional[tf.keras.layers.Layer] = None,
                virtual_batch_size: Optional[int] = None,
                momentum: Optional[float] = 0.02,
        ):
            super(TabNetModel, self).__init__()

            self.outputs = outputs
            self.n_steps = n_steps
            self.n_features = n_features
            self.feature_column = feature_column
            self.pretrained_encoder = pretrained_encoder
            self.virtual_batch_size = virtual_batch_size
            self.gamma = gamma
            self.epsilon = epsilon
            self.momentum = momentum
            self.sparsity = sparsity

            if feature_column is None:
                self.feature = tf.keras.layers.Lambda(identity)
            else:
                self.feature = feature_column

            if pretrained_encoder is None:
                self.encoder = TabNetEncoder(
                    units=outputs,
                    n_steps=n_steps,
                    n_features=n_features,
                    outputs=outputs,
                    gamma=gamma,
                    epsilon=epsilon,
                    sparsity=sparsity,
                    virtual_batch_size=self.virtual_batch_size,
                    momentum=momentum,
                )
            else:
                self.encoder = pretrained_encoder

        def forward(self, X, training=None):
            X = self.feature(X)
            prediction, encoded, importance = self.encoder(X)
            return prediction, encoded, importance

        def call(self, X, training=None):
            prediction, _, _ = self.forward(X)
            return prediction

        def transform(self, X, training=None):
            _, encoded, _ = self.forward(X)
            return encoded

        def explain(self, X, training):
            _, _, importance = self.forward(X)
            return importance
    class TabNetClassifier(TabNetModel):
        def call(self, X, training=None):
            prediction, _, _ = self.forward(X)
            if self.outputs > 1:
                return tf.keras.activations.softmax(prediction)
            else:
                return tf.keras.activations.sigmoid(prediction)
    class TabNetRegressor(TabNetModel):
        pass
    class TabNetAutoencoder(tf.keras.Model):
        def __init__(
                self,
                outputs: int = 1,
                inputs: int = 12,
                n_steps: int = 3,
                n_features: int = 8,
                gamma: float = 1.3,
                epsilon: float = 1e-8,
                sparsity: float = 1e-5,
                feature_column=None,
                virtual_batch_size=128,
                momentum=0.02):
            super(TabNetAutoencoder, self).__init__()
            self.outputs = outputs
            self.inputs = inputs
            self.n_steps = n_steps
            self.n_features = n_features
            self.feature_column = feature_column
            self.virtual_batch_size = virtual_batch_size
            self.gamma = gamma
            self.epsilon = epsilon
            self.momentum = momentum
            self.sparsity = sparsity
            if feature_column is None:
                self.feature = tf.keras.layers.Lambda(identity)
            else:
                self.feature = feature_column
            self.encoder = TabNetEncoder(
                units=outputs,
                n_steps=n_steps,
                n_features=n_features,
                outputs=outputs,
                gamma=gamma,
                epsilon=epsilon,
                sparsity=sparsity,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            self.decoder = TabNetDecoder(
                units=inputs,
                n_steps=n_steps,
                n_features=n_features,
                virtual_batch_size=self.virtual_batch_size,
                momentum=momentum,
            )
            self.bn = tf.keras.layers.BatchNormalization(virtual_batch_size=self.virtual_batch_size, momentum=momentum)
            self.do = tf.keras.layers.Dropout(0.25)

        def forward(self, X, training=None):
            X = self.feature(X)
            X = self.bn(X)
            M = self.do(tf.ones_like(X), training=training)
            D = X * M
            output, encoded, importance = self.encoder(D)
            prediction = tf.keras.activations.sigmoid(output)
            return prediction, encoded, importance, X, M

        def call(self, X, training=None):
            prediction, encoded, _, X, M = self.forward(X)
            T = X * (1 - M)
            reconstruction = self.decoder(encoded) * (1 - M) + X * M
            return reconstruction

        def transform(self, X, training=None):
            _, encoded, _, _, _ = self.forward(X)
            return encoded

        def explain(self, X, training=None):
            _, _, importance, _, _ = self.forward(X)
            return importance
    def get_feature(x, dimension=1):
        if x.dtype == np.float32:
            return tf.feature_column.numeric_column(x.name)
        else:
            return tf.feature_column.embedding_column(
                tf.feature_column.categorical_column_with_identity(x.name, num_buckets=x.max() + 1, default_value=0),
                dimension=dimension)
    def df_to_dataset(X, y, shuffle=False, batch_size=50000):
        ds = tf.data.Dataset.from_tensor_slices((dict(X.copy()), y.copy()))
        if shuffle: ds = ds.shuffle(buffer_size=len(X))
        ds = ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    feature_dim = 618
    num_decision_steps = 9
    sparse = 0.08268479989849631

    input_1 = layers.Input(shape=(n_features,), name='Input1')
    input_2 = layers.Input(shape=(n_features_2,), name='Input2')

    l = TabNetModel(outputs=206, n_features=feature_dim, # n_features + n_features_2,
                    n_steps=num_decision_steps, sparsity=sparse)(Concatenate()([input_1, input_2]))
    output = layers.Dense(n_labels, activation="sigmoid")(l)
    model = Model(inputs = [input_1, input_2], outputs = output)
    model.summary()
    model.compile(optimizer=tfa.optimizers.SWA(tf.optimizers.Adam(lr=0.001), start_averaging=9, average_period=6), loss=weighted_logloss(weights), metrics=logloss)

    input_1_1 = layers.Input(shape = (n_features,), name = 'Input1')
    input_2_1 = layers.Input(shape = (n_features_2,), name = 'Input2')
    mask_1 = layers.Dropout(0.3)((tf.ones_like(input_1_1)))
    mask_2 = layers.Dropout(0.3)((tf.ones_like(input_2_1)))

    encoded = model([input_1_1 * mask_1, input_2_1 * mask_2])

    head_4 = Sequential([
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(256, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(n_features, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        layers.Dense(n_features, activation="sigmoid")
        ],name='Head4')
    decoded = head_4(encoded)
    reconstruction_1 = decoded * (1 - mask_1) + input_1_1 * mask_1

    head_5 = Sequential([
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(256, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        tfa.layers.WeightNormalization(layers.Dense(n_features_2, kernel_initializer='lecun_normal', activation='selu')),
        layers.BatchNormalization(),
        layers.Dense(n_features_2, activation="sigmoid")
        ],name='Head5')
    decoded = head_5(encoded)
    reconstruction_2 = decoded * (1 - mask_2) + input_2_1 * mask_2

    loss = tf.reduce_mean(tf.math.abs(input_1_1 - reconstruction_1)) + tf.reduce_mean(tf.math.abs(input_2_1 - reconstruction_2))
    ae_model = Model(inputs=[input_1_1, input_2_1], outputs=[loss])
    ae_model.compile(optimizer=tf.optimizers.Adam(), loss=lambda t, y: y)

    return model, ae_model
