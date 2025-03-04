import tensorflow as tf

from official.nlp.bert.run_classifier import get_dataset_fn
from official.nlp.bert.input_pipeline import create_classifier_dataset, single_file_dataset

def test_get_dataset_fn_training_with_num_samples():
    # 修改成模拟输入的路径
    input_file_pattern = "test/dummy.tfrecord"
    seq_length = 128
    batch_size = 32
    is_training = True
    label_type = tf.int64
    include_sample_weights = False
    num_samples = 10  # 限制样本数量为10

    # 测试 get_dataset_fn 函数
    dataset_fn = get_dataset_fn(
        input_file_pattern,
        seq_length,
        batch_size,
        is_training,
        label_type,
        include_sample_weights,
        num_samples=num_samples
    )

    # 创建数据集
    dataset = dataset_fn()

    assert dataset is not None, "Dataset creation failed"

def test_create_classifier_dataset_training_with_num_samples():
    # 修改成模拟输入的路径
    input_file = "test/dummy.tfrecord" 
    num_samples = 10  # 限制样本数量为 10
    seq_length = 128
    batch_size = 8  

    # 调用 create_classifier_dataset
    dataset = create_classifier_dataset(
        file_path=input_file,
        seq_length=seq_length,
        batch_size=batch_size,
        is_training=True,
        input_pipeline_context=None,
        label_type=tf.int64,
        include_sample_weights=False,
        num_samples=num_samples
    )

    assert dataset is not None, "Dataset creation failed"
    
def test_single_file_dataset_training_with_num_samples():
    input_file = "test/dummy.tfrecord"
    num_samples = 10
    name_to_features = {
        'input_ids': tf.io.FixedLenFeature([128], tf.int64),
        'input_mask': tf.io.FixedLenFeature([128], tf.int64),
        'segment_ids': tf.io.FixedLenFeature([128], tf.int64),
        'label_ids': tf.io.FixedLenFeature([], tf.int64),
    }

    # 调用 single_file_dataset 函数，限制样本数量为 10
    dataset = single_file_dataset(input_file, name_to_features, num_samples=num_samples)

    # 验证数据集中的样本数量是否正确
    count = 0
    for _ in dataset:
        count += 1

    assert count == num_samples, f"Expected {num_samples} samples, but got {count} samples."
    
# 运行测试
if __name__ == "__main__":
    i = 0
    try:
        test_get_dataset_fn_training_with_num_samples()
        i += 1
        print(f"Test 1 passed: {i}/3")
    except Exception as e:
        print(f"Test 1 failed: {e}")
    
    try:
        test_create_classifier_dataset_training_with_num_samples()
        i += 1
        print(f"Test 2 passed: {i}/3")
    except Exception as e:
        print(f"Test 2 failed: {e}")
    
    try:
        test_single_file_dataset_training_with_num_samples()
        i += 1
        print(f"Test 3 passed: {i}/3")
    except Exception as e:
        print(f"Test 3 failed: {e}")
