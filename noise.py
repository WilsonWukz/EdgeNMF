import numpy as np
from PIL import Image
import os

def salt_pepper(X, p=0.4, r=0.3):
    """
    Add salt-and-pepper noise to an image matrix (X) ensuring the total
    proportion of corrupted pixels is exactly 'p'.

    Args:
        X (np.ndarray): Original image matrix with values in [0,1].
        p (float): Total proportion of pixels to corrupt (p).
        r (float): Fraction of corrupted pixels to set as white (salt).

    Returns:
        X_noisy (np.ndarray): Noisy image matrix.
        noise (np.ndarray): Noise matrix (1.0 for salt, -1.0 for pepper, 0.0 otherwise).
    """

    X_noisy = X.copy()
    noise = np.zeros_like(X, dtype=np.float32)

    num_pixels = X_noisy.size
    num_noise = int(num_pixels * p)

    if num_noise == 0:
        return X_noisy, noise

    all_noise_coord = np.random.choice(num_pixels, num_noise, replace=False)

    num_salt = int(num_noise * r)
    num_pepper = num_noise - num_salt

    np.random.shuffle(all_noise_coord)
    salt_coord = all_noise_coord[:num_salt]
    pepper_coord = all_noise_coord[num_salt:]

    X_noisy.flat[salt_coord] = 1.0
    noise.flat[salt_coord] = 1.0

    X_noisy.flat[pepper_coord] = 0.0
    noise.flat[pepper_coord] = -1.0

    return X_noisy, noise


def load_data_salt_pepper(root='data/CroppedYaleB', reduce=4, p=0, r=0):
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

            # Remove background images in Extended YaleB dataset.
            if fname.endswith('Ambient.pgm'):
                continue

            if not fname.endswith('.pgm'):
                continue

            # load image.
            img = Image.open(os.path.join(root, person, fname))
            img = img.convert('L')  # grey image.

            # reduce computation complexity.
            img = img.resize([s // reduce for s in img.size])

            # Convert to float
            img = np.asarray(img, dtype='float32')
            img /= 255.0

            # apply salt and pepper BEFORE L2 normalization
            img, noise_mask = salt_pepper(img, p=p, r=r)

            # then L2-normalize
            img /= (np.linalg.norm(img) + 1e-8)

            # convert image to numpy array.
            img = img.reshape((-1, 1))

            # collect data and label.
            images.append(img)
            labels.append(i)

    # concate all images and labels.
    images = np.concatenate(images, axis=1)
    labels = np.array(labels)

    return images, labels