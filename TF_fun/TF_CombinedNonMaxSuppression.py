import tensorflow as tf


boxes=tf.constant([[[[2,3,4],
                   [8,9,2],
                   [7,2,4]]]],dtype=tf.float32)
scores=tf.constant([[[2,3,8],
                     [2,6,8],
                     [6,8,9]]],dtype=tf.float32)
max_output_size_per_class=tf.constant
max_total_size=tf.constant
iou_threshold=tf.constant([])
score_threshold=tf.constant([])

Combined=tf.raw_ops.CombinedNonMaxSuppression(boxes=boxes, scores=scores,
                                              max_output_size_per_class=3, max_total_size=5,
                                              iou_threshold=0.5, score_threshold=float('-inf'))