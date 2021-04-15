import tensorflow as tf

a = tf.constant([[12, 35, 5, 7, 9]], dtype=tf.int64)
sampled_candidates,true_expected_count,sampled_expected_count = tf.nn.fixed_unigram_candidate_sampler(
    true_classes=a,
    num_true=5,#每个训练示例的目标类数
    num_sampled=3,#随机抽样的类数.
    unique=False, #确定批处理中的所有采样类是否都是唯一的.
    range_max=5,#可能的类数
    vocab_file='',
    distortion=1.0,
    num_reserved_ids=0,
    num_shards=1,#采样器可用于从原始范围的子集中进行采样,以便通过并行性加速整个计算
    shard=0,#采样器可用于从原始范围的子集中进行采样,以便通过并行性加速整个计算
    unigrams=(0.1, 0.2, 0.3, 0.1, 0.3)
)

with tf.Session() as sess:
    print(sess.run(sampled_candidates))#随机产生的采样数据
    print(sess.run(true_expected_count))#期望的各值的概率分布
    print(sess.run(sampled_expected_count))#真实产生的数据各值的概率分布
