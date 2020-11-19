import yaml
import easydict
from os.path import join


class Dataset:
    def __init__(self, path, domains, files, prefix):
        self.path = path
        self.prefix = prefix
        self.domains = domains
        self.files = [(join(path, file)) for file in files]
        self.prefixes = [self.prefix] * len(self.domains)


import argparse

parser = argparse.ArgumentParser(description='Code for *Universal Domain Adaptation*',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--config', type=str, default='config.yaml', help='/path/to/config/file')

args = parser.parse_args()
config_file = args.config
args = yaml.load(open(config_file))

save_config = yaml.load(open(config_file))

args = easydict.EasyDict(args)

dataset = None
if args.data.dataset.name == 'office':
    dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['amazon', 'dslr', 'webcam'],
        files=[
            'amazon.txt',
            'dslr.txt',
            'webcam.txt'
        ],
        prefix=args.data.dataset.root_path)
elif args.data.dataset.name == 'officehome':
    dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['Art', 'Clipart', 'Product', 'Real_World'],
        files=[
            'Art.txt',
            'Clipart.txt',
            'Product.txt',
            'Real_World.txt'
        ],
        prefix=args.data.dataset.root_path)
elif args.data.dataset.name == 'domainnet':
    dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
        files=[
            'clipart_train.txt',
            'infograph_train.txt',
            'painting_train.txt',
            'quickdraw_train.txt',
            'real_train.txt',
            'sketch_train.txt'
        ],
        prefix=args.data.dataset.root_path)
elif args.data.dataset.name == 'visda2017':
    dataset = Dataset(
        path=args.data.dataset.root_path,
        domains=['train', 'validation'],
        files=[
            'train_list.txt',
            'validation_list.txt'
        ],
        prefix=args.data.dataset.root_path)
    dataset.prefixes = [join(dataset.path, 'train'), join(dataset.path, 'validation')]
else:
    raise Exception(f'dataset {args.data.dataset.name} not supported!')

source_domain_name = dataset.domains[args.data.dataset.source]
target_domain_name = dataset.domains[args.data.dataset.target]
source_file = dataset.files[args.data.dataset.source]
target_file = dataset.files[args.data.dataset.target]
