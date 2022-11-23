from PIL import Image


def _get_img_from_path(img_path, transform=None):
    with open(img_path, 'rb') as f:
        img = Image.open(f).convert('RGB')
    if transform is not None:
        img = transform(img)
    return img
