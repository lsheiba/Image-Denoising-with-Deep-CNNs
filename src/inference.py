import argparse
import time

import cv2
import numpy as np
import torch
import torch.quantization.quantize_fx as quantize_fx
from torch.quantization.fuse_modules import fuse_known_modules

import model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', default=6, type=int)
    parser.add_argument('-C', default=64, type=int)
    parser.add_argument('--image')
    parser.add_argument('--output')
    parser.add_argument(
        '--quantize',
        choices=[None, 'dynamic', 'static', 'fx_dynamic', 'fx_static'],
        default=None
    )
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--model', required=True)

    return parser.parse_args()


#added variable model loading for QAT
def load_model(path, model_class=model.DUDnCNN, D=6, C=64, device=torch.device('cpu')):
    net = model_class(D, C)
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    if 'QAT' not in checkpoint:
        net.load_state_dict(checkpoint['Net'])
    net.eval()

    return net.to(device), checkpoint.get('QAT')


def img_to_tensor(img, device):
    tensor = torch.FloatTensor(img).to(device)
    tensor = tensor.permute([2, 0, 1]) / 255.
    tensor = (tensor - 0.5) / 0.5

    return tensor.unsqueeze(0)


def tensor_to_img(tensor):
    tensor = tensor[0].permute([1, 2, 0])
    tensor = (tensor * 0.5 + 0.5) * 255
    tensor = tensor.clamp(0, 255)
    return tensor.cpu().numpy().astype(np.uint8)


def quantize_model(quantize_type, model, input_example=None):
    if quantize_type == 'dynamic':
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Conv2d},
            dtype=torch.qint8
        )
    elif quantize_type == 'static':
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        for i in range(len(model.bn)):
            conv, bn = model.conv[i+1], model.bn[i]
            conv_new, bn_new = fuse_known_modules([conv, bn])
            setattr(model.conv, str(i+1), conv_new)
            setattr(model.bn, str(i), bn_new)
        model_fp32_fused = model
        model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)
        if input_example is not None:
            model_fp32_prepared(input_example)
        model = torch.quantization.convert(model_fp32_prepared)
    elif quantize_type == 'fx_dynamic':
        qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}
        # prepare
        model_prepared = quantize_fx.prepare_fx(model, qconfig_dict)
        # no calibration needed when we only have dynamici/weight_only quantization
        # quantize
        model = quantize_fx.convert_fx(model_prepared)
    elif quantize_type == 'fx_static':
        # qconfig_dict = {"": torch.quantization.get_default_qconfig('qnnpack')}
        qconfig_dict = {"": torch.quantization.get_default_qconfig('fbgemm')}
        # prepare
        model_prepared = quantize_fx.prepare_fx(model, qconfig_dict)
        # calibrate (not shown)
        if input_example is not None:
            model_prepared(input_example)
        # quantize
        model = quantize_fx.convert_fx(model_prepared)

    return model


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    denoise = load_model(args.model, args.D, args.C, device=device)

    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    small = cv2.resize(img, (720, 720), interpolation=cv2.INTER_AREA)

    tensor = img_to_tensor(small, device)

    if args.quantize:
        print('Quantize model...')
        denoise = quantize_model(args.quantize, denoise, input_example=tensor)
        print('Done.')

    t = time.time()
    with torch.no_grad():
        output = denoise(tensor)

    print(f'Elapsed: {(time.time() - t) * 1000:.2f}ms')
    output = tensor_to_img(output)
    combined = np.hstack([small, output])

    if args.show:
        cv2.imshow('Image', combined[:, :, ::-1])
        cv2.waitKey(0)

    if args.output:
        cv2.imwrite(args.output, combined[:, :, ::-1])
        print(f'Image saved to {args.output}.')


if __name__ == '__main__':
    main()
