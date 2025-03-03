# 用户实验介绍

## 环境配置
请通过以下命令配置环境，用于测试编辑结果：
```bash
conda create --name user_study python=3.11 -y
conda activate user_study
pip install tensorflow==2.18.0
pip install gin-config
pip install tensorflow_addons
```

## 任务介绍
我们希望添加一个新的参数 `num_samples` 来控制加载的训练集样本数量。首个编辑发生在 [`official/nlp/bert/run_classifier.py`](official/nlp/bert/run_classifier.py) 文件中，我们为函数 `get_dataset_fn` 添加了新的参数 `num_samples`，由尚未定义的 `FLAGS.train_data_size` 传入，如图所示：

![image](./images/init_edit.png)

请你在完成所示修改后，继续对项目进行 **另外 8 处** 修改。你可以打开源代码管理工具，查看修改的数量，确保你确实完成了 8 处修改，如图所示：

![image](./images/git_diff.png)

如果你觉得修改的次数不够，可以继续修改，直到实现编辑目标（即通过 [验证修改](#验证修改) 中的测试）


## 温馨提示
1. `FLAGS.train_data_size` 的默认值是 `None`，可能被设置的比实际训练集样本数量大，请注意这一点。
2. Tensorflow 项目使用 `tf.data.TFRecordDataset` 来加载数据集:
    ```python
    dataset = tf.data.TFRecordDataset(file_path)
    ```
3. Tensorflow 提供了 `tf.data.Dataset.take` 方法来控制数据集的样本数量：
    ```python
    dataset = dataset.take(num_samples)
    ```
4. 为了优化数据集加载效率，请在从文件加载数据集之后，立刻控制数据集的样本数量。

## 验证修改
请运行以下命令验证修改：
```bash
python -m test.test
```

当修改正确时，测试会输出以下信息：