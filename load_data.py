import numpy as np
import os
from PIL import Image

def load_data(root='data/CroppedYaleB', reduce=4):
    """
    Load ORL (or Extended YaleB) dataset to numpy array.

    Args:
        root: path to dataset.
        reduce: scale factor for zooming out images.

    """
    images, labels = [], []

    for i, person in enumerate(sorted(os.listdir(root))):

        if not os.path.isdir(os.path.join(root, person)):
            continue

        for fname in os.listdir(os.path.join(root, person)):

            if fname.endswith('Ambient.pgm'):
                continue

            if not fname.endswith('.pgm'):
                continue

            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L')

            img = img.resize([s // reduce for s in img.size])

            img = np.asarray(img, dtype='float32')
            # Normalize pixel values to [0,1]
            img = img / 255

            # Mean centering or variance scaling
            img /= np.linalg.norm(img) + 1e-8

            # convert image to numpy array.
            img = img.reshape((-1, 1))

            # collect data and label.
            images.append(img)
            labels.append(i)

    # concate all images and labels.
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)

    return images, labels