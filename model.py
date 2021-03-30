
from __future__ import division, print_function

import tensorflow as tf
slim = tf.contrib.slim

from utils.layer_utils import conv2d, darknet53_body, yolo_block, upsample_layer

class yolov3(object):
    '''
    yolov3模型，用于车牌检测和识别
    '''
    def __init__(self, class_num, anchors, use_label_smooth=False, use_focal_loss=True, batch_norm_decay=0.999, weight_decay=5e-4, use_static_shape=True):

        self.class_num = class_num
        self.anchors = anchors
        self.batch_norm_decay = batch_norm_decay
        self.use_label_smooth = use_label_smooth
        self.use_focal_loss = use_focal_loss
        self.weight_decay = weight_decay
        self.use_static_shape = use_static_shape

    def forward(self, inputs, is_training=False, reuse=False):
        '''
        向前传播算法，接收图像矩阵输入，输出置信度和坐标框
        '''
        self.img_size = tf.shape(inputs)[1:3]

        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None, 
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d], 
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = darknet53_body(inputs)

                with tf.variable_scope('yolov3_head'):
                    inter1, net = yolo_block(route_3, 512)
                    feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1, route_2.get_shape().as_list() if self.use_static_shape else tf.shape(route_2))
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yolo_block(concat1, 256)
                    feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, route_1.get_shape().as_list() if self.use_static_shape else tf.shape(route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block(concat2, 128)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            return feature_map_1, feature_map_2, feature_map_3

    def reorg_layer(self, feature_map, anchors):
        '''
        区域建议网络，提取ROI
        '''
        grid_size = feature_map.get_shape().as_list()[1:3] if self.use_static_shape else tf.shape(feature_map)[1:3]  # [13, 13]
        ratio = tf.cast(self.img_size / grid_size, tf.float32)

        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)

        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        box_centers = box_centers + x_y_offset
        box_centers = box_centers * ratio[::-1]

        box_sizes = tf.exp(box_sizes) * rescaled_anchors

        box_sizes = box_sizes * ratio[::-1]

        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        return x_y_offset, boxes, conf_logits, prob_logits


    def predict(self, feature_maps):
        '''
        CNN检测器模块，输入为特征层，输出为预测值
        '''
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = x_y_offset.get_shape().as_list()[:2] if self.use_static_shape else tf.shape(x_y_offset)[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])

            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)
        

        boxes = tf.concat(boxes_list, axis=1)
        confs = tf.concat(confs_list, axis=1)
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs
    
    def loss_layer(self, feature_map_i, y_true, anchors):
        '''
        损失函数，用于反向传播算法
        '''
        grid_size = tf.shape(feature_map_i)[1:3]
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)

        object_mask = y_true[..., 4:5]

        ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        def loop_cond(idx, ignore_mask):
            return tf.less(idx, tf.cast(N, tf.int32))
        def loop_body(idx, ignore_mask):
            # shape: [13, 13, 3, 4] & [13, 13, 3]  ==>  [V, 4]
            # V: num of true gt box of each image in a batch
            valid_true_boxes = tf.boolean_mask(y_true[idx, ..., 0:4], tf.cast(object_mask[idx, ..., 0], 'bool'))
            # shape: [13, 13, 3, 4] & [V, 4] ==> [13, 13, 3, V]
            iou = self.box_iou(pred_boxes[idx], valid_true_boxes)
            # shape: [13, 13, 3]
            best_iou = tf.reduce_max(iou, axis=-1)
            # shape: [13, 13, 3]
            ignore_mask_tmp = tf.cast(best_iou < 0.5, tf.float32)
            # finally will be shape: [N, 13, 13, 3]
            ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
            return idx + 1, ignore_mask
        _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        # shape: [N, 13, 13, 3, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # shape: [N, 13, 13, 3, 2]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        # mix_up weight
        # [N, 13, 13, 3, 1]
        mix_w = y_true[..., -1:]
        # shape: [N, 13, 13, 3, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale * mix_w) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=pred_conf_logits)
        conf_loss = conf_loss_pos + conf_loss_neg
        if self.use_focal_loss:
            alpha = 1.0
            gamma = 2.0
            focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf_logits)), gamma)
            conf_loss *= focal_mask
        conf_loss = tf.reduce_sum(conf_loss * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        if self.use_label_smooth:
            delta = 0.01
            label_target = (1 - delta) * y_true[..., 5:-1] + delta * 1. / self.class_num
        else:
            label_target = y_true[..., 5:-1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target, logits=pred_prob_logits) * mix_w
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss
    

    def box_iou(self, pred_boxes, valid_true_boxes):
        '''
        计算两个坐标框的重叠度（iou）
        '''
        # [13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # shape: [13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # [V, 2]
        true_box_xy = valid_true_boxes[:, 0:2]
        true_box_wh = valid_true_boxes[:, 2:4]

        # [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = tf.expand_dims(true_box_area, axis=0)

        # [13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

        return iou

    
    def compute_loss(self, y_pred, y_true):
        '''
        计算不同的损失函数，包括：总损失，xy坐标损失，wh长/高损失，置信度损失和类别损失
        '''
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]

        # calc loss in 3 scales
        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]
