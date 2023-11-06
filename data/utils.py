from torchvision import transforms


def standardize(img):
    # standarized_value=(raw_value-μ)/σ
    return (img - img.mean()) / img.std()

def normalize(img):
    # normalized_value=(raw_value-min_value)/(max_value-min_value)
    return (img - img.min()) / (img.max() - img.min())


def std_norm(img):
    # standardize the image
    img = standardize(img)
    # normalize the image
    img = normalize(img)
    return img