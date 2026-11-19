# Liveness Eval Lab

2019–2020 年，我做了面向端侧的静默活体检测模型剪枝和知识蒸馏。我以 MobileFaceNet 为骨干，比较不同的通道配置。其中一个紧凑模型相对公开的 MiniFASNet V1，在 49 个通道配置项里收窄了 18 项，其余 31 项沿用原宽度。

训练学生网络时，我用过两种蒸馏方法。常规知识蒸馏让学生结合真实标签学习教师给出的类别概率。关系蒸馏让学生学习教师对一批样本的判断关系，包括哪些样本彼此接近，以及它们在输出空间中的相对方向。[剪枝与蒸馏实验笔记](docs/experiments.md)

## 运行剪枝模型与蒸馏训练

`compact-a` 和 `compact-b` 各用 49 个通道配置项描述骨干。输入为 80×80 RGB 裁剪，默认输出三个类别的 logits。下面的命令用合成输入跑通一次蒸馏，保存的学生权重可以直接导出为 Caffe 和 NCNN。[训练数据、教师接口和三端转换示例](docs/train-export.md)

```sh
python3 -m pip install '.[train]'
python3 -m liveness_lab.distill --method kd --preset compact-a --output student.pt
```

## 配套评估工具

仓库附有一个离线工具，可直接评估 [Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) 输出的逐模型真人概率。在仓库根目录使用 Python 3.9 及以上运行示例即可。

```sh
python3 -m liveness_lab evaluate examples/ensemble.csv --target-fpr 0.10 --max-size 2
```

<details>
<summary>查看示例结果、输入格式和其他命令</summary>


示例有两个模型，程序分别评估两个单模型和一个双模型组合。下表是命令的实际输出摘要。数据由手工构造，全部比例都能从随附 CSV 逐行复算。

| 开发集选出的方案 | 冻结阈值 | 开发集真人通过 | 开发集攻击通过 | 测试集真人通过 | 测试集攻击通过 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 最佳单模型 `context` | 0.94 | 1/4 | 0/4 | 0/4 | 0/4 |
| `context` + `detail` | 0.69 | 4/4 | 0/4 | 3/4 | 1/4 |

这里的 FPR 是攻击样本通过的比例，TPR 是真人样本通过的比例。开发集只有 4 个攻击样本，一次误放就是 25%。设置 10% 上限时，开发集只能选零误放的阈值。测试集上的 `1/4` 保留在报告里，读者能直接看见阈值换一批样本后的表现。

### 接入自己的模型

每行对应同一条采集样本，每个模型占一列。`sample_id` 唯一，`group_id` 用于把同一人或相关采集归在一个 split 内。建议在本地生成替代编号。先按样本 ID 对齐各模型输出，缺失分数应在生成 CSV 时单独处理。工具会检查跨 split 的组、重复的样本 ID、空列和越界分数。

```csv
sample_id,group_id,split,label,context,detail
d01,g05,dev,live,0.94,0.58
```

要在上游推理环境导入本工具，可先在本仓库根目录执行 `python3 -m pip install .`。在上游 `test.py` 的模型循环内，每个裁剪分别经过 `predict`。保留该次调用返回的概率向量，用类别索引 `1` 取出真人概率，再按模型写入各自的列。上游最后显示的是获胜类别的置信度，假脸样本的这个数也可能很高。把它当真人概率会颠倒分数含义。[上游推理入口](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/b6d5f04ad78778917853b25c778acef6d5626d15/test.py#L59)

```python
from liveness_lab.data import live_probability

probabilities = model_test.predict(img, model_path)
score = live_probability(probabilities[0], live_index=1)
```

`evaluate` 在 `dev` 上对等权平均后的分数选阈值，按 `score >= threshold` 判为真人。先在 FPR 上限内取最高 TPR，同 TPR 时取更低 FPR，再依次优先较少模型和字典序较前的名称。`null` 阈值表示全部拒绝。`dev_nearest_fpr` 另给出最接近目标 FPR 的开发集结果。工具只把严格上限选出的模型和阈值带入 `test`。反复查看测试结果并据此改模型后，需要新的留出集。

### 难例与转换检查

```sh
python3 -m liveness_lab mine examples/ensemble.csv --model context --per-class 1
python3 -m liveness_lab parity examples/parity.csv --reference reference --converted converted --threshold 0.70
python3 -m unittest discover -s tests -v
```

`mine` 只读取 `train`，分别列出分数最低的真人和最高的攻击样本，供回看标签和采集质量。示例返回 `tr02`、`tr03`。运行自己的数据时，返回的 ID 适合留在本地。

`parity` 在 `test` 上比较同一批输入的两个后端，报告平均、最大绝对分差，以及越过阈值的方向。示例最大分差仅 0.01，却有一个真人从通过变为拒绝，一个攻击样本从拒绝变为通过。要复查转换误差，两份分数应来自同一批输入、同一套裁剪与预处理。

</details>

## 许可与来源

本仓库的 Python 实现、文档与手工示例采用 MIT 许可。示例里的模型名称、样本编号、分数和阈值均为演示用途。上游代码和权重由上游项目分发，其仓库附有 Apache-2.0 许可证。[本仓库许可](LICENSE) · [上游许可](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/b6d5f04ad78778917853b25c778acef6d5626d15/LICENSE)
