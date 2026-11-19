"""Export the compact classifier to Caffe and optionally NCNN."""

import argparse
import operator
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, text_format
from grpc_tools import protoc
from torch import nn
from torch.fx import symbolic_trace

from .compact import load_model


def net_type(schema, work):
    schema = Path(schema).resolve(strict=True)
    descriptor = work / 'schema.pb'
    status = protoc.main(['protoc', f'-I{schema.parent}', f'--descriptor_set_out={descriptor}', schema.name])
    if status:
        raise ValueError('could not compile the Caffe protobuf schema')
    pool = descriptor_pool.DescriptorPool()
    for definition in descriptor_pb2.FileDescriptorSet.FromString(descriptor.read_bytes()).file:
        pool.Add(definition)
    return message_factory.GetMessageClass(pool.FindMessageTypeByName('caffe.NetParameter'))


def add_weights(layer, tensors):
    for tensor in tensors:
        blob = layer.blobs.add()
        blob.shape.dim.extend(tensor.shape)
        blob.data.extend(tensor.detach().cpu().reshape(-1).tolist())


def write_caffe(model, schema, work):
    graph = symbolic_trace(model.eval())
    net_class = net_type(schema, work)
    topology = net_class(name='compact_liveness')
    input_layer = topology.layer.add(name='data', type='Input', top=['data'])
    input_layer.input_param.shape.add().dim.extend((1, 3, 80, 80))
    parameters = net_class(name='compact_liveness')
    names = {}
    for node in graph.graph.nodes:
        if node.op == 'placeholder':
            names[node] = 'data'
            continue
        if node.op == 'output':
            if names[node.args[0]] != 'logits':
                raise ValueError('unexpected classifier output')
            continue
        bottoms = [names[parent] for parent in node.all_input_nodes]
        module = graph.get_submodule(node.target) if node.op == 'call_module' else None
        if isinstance(module, (nn.Dropout, nn.BatchNorm1d, nn.BatchNorm2d)):
            if not isinstance(module, nn.Dropout):
                parent = node.args[0]
                if parent.op != 'call_module' or not isinstance(graph.get_submodule(parent.target), (nn.Linear, nn.Conv2d)):
                    raise ValueError('expected a foldable batch normalization')
            names[node] = bottoms[0]
            continue
        layer = topology.layer.add(name=node.name, bottom=bottoms, top=[node.name])
        names[node] = node.name
        tensors = []
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            users = list(node.users)
            successor = graph.get_submodule(users[0].target) if len(users) == 1 and users[0].op == 'call_module' else None
            if isinstance(successor, (nn.BatchNorm1d, nn.BatchNorm2d)):
                fuse = nn.utils.fuse_conv_bn_eval if isinstance(module, nn.Conv2d) else nn.utils.fuse_linear_bn_eval
                module = fuse(module, successor)
            tensors = [module.weight] if module.bias is None else [module.weight, module.bias]
            if isinstance(module, nn.Conv2d):
                layer.type = 'Convolution'
                params = layer.convolution_param
                params.num_output, params.group = module.out_channels, module.groups
                params.kernel_h, params.kernel_w = module.kernel_size
                params.stride_h, params.stride_w = module.stride
                params.pad_h, params.pad_w = module.padding
            else:
                layer.type = 'InnerProduct'
                params = layer.inner_product_param
                params.num_output = module.out_features
            params.bias_term = module.bias is not None
        elif isinstance(module, nn.PReLU):
            layer.type = 'PReLU'
            tensors = [module.weight]
        elif isinstance(module, nn.Flatten):
            layer.type = 'Flatten'
        elif node.op == 'call_function' and node.target is operator.add:
            layer.type = 'Eltwise'
            layer.eltwise_param.operation = 1
        else:
            raise ValueError(f'unsupported operation {node.target}')
        if tensors:
            weights = parameters.layer.add(name=node.name, type=layer.type)
            add_weights(weights, tensors)
    (work / 'model.prototxt').write_text(text_format.MessageToString(topology), encoding='utf-8')
    (work / 'model.caffemodel').write_bytes(parameters.SerializeToString())


def verify(model, directory, include_ncnn):
    import cv2
    import numpy as np

    caffe_net = cv2.dnn.readNetFromCaffe(str(directory / 'model.prototxt'), str(directory / 'model.caffemodel'))
    if include_ncnn:
        import ncnn

        mobile = ncnn.Net()
        mobile.opt.num_threads = 1
        mobile.opt.use_packing_layout = False
        mobile.opt.use_fp16_storage = False
        if mobile.load_param(str(directory / 'model.param')) or mobile.load_model(str(directory / 'model.bin')):
            raise ValueError('NCNN could not load the exported model')
    generator = torch.Generator().manual_seed(31)
    errors = {'caffe': 0.0}
    if include_ncnn:
        errors['ncnn'] = 0.0
    for _ in range(3):
        inputs = torch.randn((1, 3, 80, 80), generator=generator)
        with torch.no_grad():
            reference = model(inputs).numpy().reshape(-1)
        caffe_net.setInput(inputs.numpy())
        outputs = {'caffe': caffe_net.forward().reshape(-1)}
        if include_ncnn:
            with mobile.create_extractor() as extractor:
                if extractor.input('data', ncnn.Mat(inputs[0].numpy()).clone()):
                    raise ValueError('NCNN rejected the input')
                status, result = extractor.extract('logits')
                if status:
                    raise ValueError('NCNN extraction failed')
                outputs['ncnn'] = np.array(result).reshape(-1)
        for backend, logits in outputs.items():
            np.testing.assert_allclose(logits, reference, rtol=1e-4, atol=1e-5, equal_nan=False)
            errors[backend] = max(errors[backend], float(np.max(np.abs(logits - reference))))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--caffe-proto', required=True)
    parser.add_argument('--out', type=Path, required=True, help='new output directory')
    parser.add_argument('--caffe2ncnn', help='path to the NCNN converter executable')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output directory already exists')
    torch.set_num_threads(1)
    model = load_model(args.checkpoint).eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.out.parent) as temporary:
        work = Path(temporary) / 'export'
        work.mkdir()
        write_caffe(model, args.caffe_proto, work)
        (work / 'schema.pb').unlink()
        if args.caffe2ncnn:
            subprocess.run([args.caffe2ncnn, *(str(work / name) for name in
                            ('model.prototxt', 'model.caffemodel', 'model.param', 'model.bin'))], check=True)
            if not all((work / name).stat().st_size for name in ('model.param', 'model.bin')):
                raise ValueError('empty NCNN output')
        if args.verify:
            print('max_abs_logit_error', verify(model, work, bool(args.caffe2ncnn)))
        work.rename(args.out)


if __name__ == '__main__':
    main()
