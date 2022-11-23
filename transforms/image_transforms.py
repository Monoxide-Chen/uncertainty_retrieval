from torchvision import transforms

IMAGENET_STATS = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


def get_train_transform(transform_config: dict):
    use_transform = transform_config['use_transform']
    img_size = transform_config['img_size']

    if use_transform:
        return transforms.Compose([transforms.RandomResizedCrop(size=img_size, scale=(0.75, 1.33)),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(**IMAGENET_STATS)])

    return transforms.Compose([transforms.Resize((img_size, img_size)),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize(**IMAGENET_STATS)])


def get_val_transform(transform_config: dict):
    img_size = transform_config['img_size']

    return transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(),
                               transforms.Normalize(**IMAGENET_STATS)])


def image_transform_factory(config: dict):
    return {
        'train': get_train_transform(config),
        'val': get_val_transform(config)
    }
