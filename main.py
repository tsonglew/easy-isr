import os
import argparse

from utils import handle_dir, load_model, resize_pics


parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', required=False, default=os.path.join(os.path.abspath('.'), 'test_inputs'))
parser.add_argument('--output_dir', required=False, default=os.path.join(os.path.abspath('.'), 'test_outputs'))
parser.add_argument('--mode', required=False, default='isr')
parser.add_argument('--weights', required=False, default=os.path.join(
    os.path.abspath('.'), 
    'weights',
    'rdn-C6-D20-G64-G064-x2', 
    'ArtefactCancelling', 
    'rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5'
    ))
parser.add_argument('--width', required=False, type=int, default=1920)
parser.add_argument('--height', required=False, type=int, default=1080)
parser.add_argument('--resize', required=False, type=bool)
parser.add_argument('--fixed_proportion', required=False, type=bool)
parser.add_argument('--padding_size', required=False, type=int, default=8)
parser.add_argument('--by_patch_of_size', required=False, type=int, default=None)


if __name__ == '__main__':
    print(args.fixed_proportion)
    if args.mode == 'isr':
        rdn = load_model(args.weights)
        handle_dir(args.input_dir, args.output_dir, rdn, args.width, args.height, args.resize, args.fixed_proportion, args.padding_size, args.by_patch_of_size)
    elif args.mode == 'resize':
        resize_pics(args.input_dir, (args.height, args.width), args.fixed_proportion)

    