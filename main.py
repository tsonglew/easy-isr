import os
import argparse

from utils import handle_dir, load_model


parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', required=False, default=os.path.join(os.path.abspath('.'), 'test_inputs'),  help='')
parser.add_argument('--output_dir', required=False, default=os.path.join(os.path.abspath('.'), 'test_outputs'), help='')
parser.add_argument('--weights', required=False, default=os.path.join(
    os.path.abspath('.'), 
    'weights',
    'rdn-C6-D20-G64-G064-x2', 
    'ArtefactCancelling', 
    'rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5'
    ), help='')
parser.add_argument('--width', required=False, type=int, default=960, help='')
parser.add_argument('--height', required=False, type=int, default=540, help='')
parser.add_argument('--fixed_proportion', required=False, type=bool, default=True, help='')


if __name__ == '__main__':
    args = parser.parse_args()
    rdn = load_model(args.weights)
    handle_dir(args.input_dir, args.output_dir, rdn, args.width, args.height, args.fixed_proportion)
    