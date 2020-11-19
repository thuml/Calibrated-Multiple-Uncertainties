from config import *
from easydl import *
from lib import *
from collections import Counter
from torchvision.transforms.transforms import *
from torch.utils.data import DataLoader, WeightedRandomSampler


'''
assume classes across domains are the same.
[0 1 ..................................................................... N - 1]
|----common classes --||----source private classes --||----target private classes --|
'''
a, b, c = args.data.dataset.n_share, args.data.dataset.n_source_private, args.data.dataset.n_total
c = c - a - b
common_classes = [i for i in range(a)]
source_private_classes = [i + a for i in range(b)]
target_private_classes = [i + a + b for i in range(c)]

source_classes = common_classes + source_private_classes
target_classes = common_classes + target_private_classes

train_transform = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.2, resample=Image.BICUBIC,
                 fillcolor=(255, 255, 255)),
    CenterCrop(224),
    RandomGrayscale(p=0.1),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform2 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    MyRandomPerspective(),
    FiveCrop(224),
    Lambda(lambda crops: crops[0]),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform3 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.2, resample=Image.BICUBIC,
                 fillcolor=(255, 255, 255)),
    FiveCrop(224),
    Lambda(lambda crops: crops[1]),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform4 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1, resample=Image.BICUBIC,
                 fillcolor=(255, 255, 255)),
    MyRandomPerspective(),
    FiveCrop(224),
    Lambda(lambda crops: crops[2]),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

train_transform5 = Compose([
    Resize(256),
    RandomHorizontalFlip(),
    MyRandomPerspective(),
    FiveCrop(224),
    Lambda(lambda crops: crops[3]),
    RandomGrayscale(p=0.5),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

test_transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

source_train_ds = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                  transform=train_transform, filter=(lambda x: x in source_classes))
source_train_ds2 = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                   transform=train_transform2, filter=(lambda x: x in source_classes))
source_train_ds3 = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                   transform=train_transform3, filter=(lambda x: x in source_classes))
source_train_ds4 = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                   transform=train_transform4, filter=(lambda x: x in source_classes))
source_train_ds5 = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                   transform=train_transform5, filter=(lambda x: x in source_classes))
source_test_ds = FileListDataset(list_path=source_file, path_prefix=dataset.prefixes[args.data.dataset.source],
                                 transform=test_transform, filter=(lambda x: x in source_classes))
target_train_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                  transform=train_transform, filter=(lambda x: x in target_classes))
target_test_ds = FileListDataset(list_path=target_file, path_prefix=dataset.prefixes[args.data.dataset.target],
                                 transform=test_transform, filter=(lambda x: x in target_classes))

classes = source_train_ds.labels
freq = Counter(classes)
class_weight = {x: 1.0 / freq[x] if args.data.dataloader.class_balance else 1.0 for x in freq}

source_weights = [class_weight[x] for x in source_train_ds.labels]
sampler = WeightedRandomSampler(source_weights, len(source_train_ds.labels))

source_train_dl = DataLoader(dataset=source_train_ds, batch_size=args.data.dataloader.batch_size,
                             sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
source_train_dl2 = DataLoader(dataset=source_train_ds2, batch_size=args.data.dataloader.batch_size,
                              sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
source_train_dl3 = DataLoader(dataset=source_train_ds3, batch_size=args.data.dataloader.batch_size,
                              sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
source_train_dl4 = DataLoader(dataset=source_train_ds4, batch_size=args.data.dataloader.batch_size,
                              sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)
source_train_dl5 = DataLoader(dataset=source_train_ds5, batch_size=args.data.dataloader.batch_size,
                              sampler=sampler, num_workers=args.data.dataloader.data_workers, drop_last=True)

source_test_dl = DataLoader(dataset=source_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                            num_workers=1, drop_last=False)
target_train_dl = DataLoader(dataset=target_train_ds, batch_size=args.data.dataloader.batch_size, shuffle=True,
                             num_workers=args.data.dataloader.data_workers, drop_last=True)
target_test_dl = DataLoader(dataset=target_test_ds, batch_size=args.data.dataloader.batch_size, shuffle=False,
                            num_workers=1, drop_last=False)
