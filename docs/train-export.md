# 紧凑模型训练与部署

公开 MiniFASNet 提供 V1、V2 的通道配置，上游训练入口用分类损失和傅里叶特征图损失训练模型。这里补充两个紧凑配置、教师学生训练，以及从学生权重导出 Caffe 和 NCNN 的工具。[上游模型定义](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/b6d5f04ad78778917853b25c778acef6d5626d15/src/model_lib/MiniFASNet.py#L265) · [上游训练目标](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/b6d5f04ad78778917853b25c778acef6d5626d15/src/train_main.py#L107)

## 剪枝配置

`channel_config` 返回剪枝后固定的通道数。骨干包含逐点卷积、逐通道卷积和残差块，末端使用 5×5 逐通道卷积聚合空间特征。默认分类头把 512 维特征映射到 128 维，再输出三个类别的 logits。

| 配置 | 中段下采样块的扩展通道 | 末段下采样块的扩展通道 | 可训练参数量 |
| --- | ---: | ---: | ---: |
| `compact-a` | 180 | 52 | 273,058 |
| `compact-b` | 154 | 103 | 281,497 |

```python
from liveness_lab.compact import CompactNet, channel_config

model = CompactNet('compact-a', classes=3)
print(channel_config('compact-a'))
print(sum(parameter.numel() for parameter in model.parameters()))
```

## KD 与 RKD

下面的 CPU 示例使用固定种子的随机教师和合成张量，可检查反向传播、权重保存与后续导出。

```sh
python3 -m pip install '.[train]'
python3 -m liveness_lab.distill --method kd --preset compact-a --output student-a.pt
python3 -m liveness_lab.distill --method rkd --preset compact-b --output student-b.pt
```

训练自己的模型时，`--teacher` 接受本项目格式的已训练教师权重，包含 `preset`、`classes`、`state_dict`。`--data` 接受由 `torch.save` 写入的字典。其中 `images` 是经过相同预处理的 `float32` 张量 `[N,3,80,80]`，`labels` 是从零开始的 `int64` 类别索引 `[N]`。教师、学生使用相同的类别顺序。接入其他骨干的教师或自定义加载器，可直接调用 `train_epoch(student, teacher, batches, optimizer, method)`，两个模型和输入应在同一设备上。

```sh
python3 -m liveness_lab.distill --method kd --data crops.pt --teacher teacher.pt --epochs 5 --output student.pt
```

KD 组合真实标签交叉熵和软目标 KL 散度。RKD 使用距离和角度两项 Smooth L1 损失，角度权重为距离的两倍。教师以评估模式输出目标。RKD 的角度张量随批大小呈三次方增长，示例的批大小为 4。

## PyTorch → Caffe → NCNN

导出器根据已训练学生的计算图生成 Caffe 网络定义和权重，把推理态批归一化折进前置卷积或全连接层，保留逐通道 PReLU 和残差相加。卷积核的高和宽分别写入 Caffe 字段。支持范围为本项目两个紧凑模型，输出节点名为 `logits`。

在使用 Python 3.11 的环境安装可选依赖，把官方 `caffe2ncnn` 可执行文件加入 `PATH`。格式定义由 BVLC Caffe 提供，转换器由 NCNN 提供，两者均可单独获取。[Caffe 格式定义](https://github.com/BVLC/caffe/blob/1.0/src/caffe/proto/caffe.proto) · [NCNN 转换器源码](https://github.com/Tencent/ncnn/tree/20240410/tools/caffe)

```sh
python3 -m pip install '.[export,verify]'
curl -fL https://raw.githubusercontent.com/BVLC/caffe/1.0/src/caffe/proto/caffe.proto -o caffe.proto
python3 -m liveness_lab.export --checkpoint student-a.pt --caffe-proto caffe.proto --out exported-a --caffe2ncnn caffe2ncnn --verify
```

请选择新的输出目录。导出和校验成功后，目录包含 `model.prototxt`、`model.caffemodel`、`model.param`、`model.bin`。只需要 Caffe 时可省略 `--caffe2ncnn`。下游需要真人概率时，先对 `logits` 做 softmax，再取训练标签中真人对应的类别。

`--verify` 把同样的三份随机输入分别交给 PyTorch、OpenCV DNN 的 Caffe 接口和 NCNN CPU，逐元素比较分类 logits。容差为 `atol=1e-5`、`rtol=1e-4`。下表记录合成训练示例的一次端到端运行，数值为相对 PyTorch 的最大绝对误差。

| 学生 | Caffe / OpenCV | NCNN |
| --- | ---: | ---: |
| `compact-a`，KD | 1.42e-8 | 5.59e-9 |
| `compact-b`，RKD | 8.38e-9 | 1.06e-8 |

校验环境使用 PyTorch 2.13、OpenCV 4.14、NCNN 的 `20240410` 转换器和 `20260526` Python 运行时。真人通过率与攻击误报率应在独立的带标签数据上评估。[评估工具用法](../README.md)
