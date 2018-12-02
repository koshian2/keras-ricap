import numpy as np

def ricap(image_batch, label_batch, beta=0.3, use_same_random_value_on_batch=True):
    # if use_same_random_value_on_batch = True : same as the original paper
    assert image_batch.shape[0] == label_batch.shape[0]
    assert image_batch.ndim == 4
    batch_size, image_y, image_x = image_batch.shape[:3]

    # crop_size w, h from beta distribution
    if use_same_random_value_on_batch:
        w_dash = np.random.beta(beta, beta) * np.ones(batch_size)
        h_dash = np.random.beta(beta, beta) * np.ones(batch_size)
    else:
        w_dash = np.random.beta(beta, beta, size=(batch_size))
        h_dash = np.random.beta(beta, beta, size=(batch_size))
    w = np.round(w_dash * image_x).astype(np.int32)
    h = np.round(h_dash * image_y).astype(np.int32)

    # outputs
    output_images = np.zeros(image_batch.shape)
    output_labels = np.zeros(label_batch.shape)


    def create_masks(start_xs, start_ys, end_xs, end_ys):
        mask_x = np.logical_and(np.arange(image_x).reshape(1,1,-1,1) >= start_xs.reshape(-1,1,1,1), 
                                np.arange(image_x).reshape(1,1,-1,1) < end_xs.reshape(-1,1,1,1))
        mask_y = np.logical_and(np.arange(image_y).reshape(1,-1,1,1) >= start_ys.reshape(-1,1,1,1),
                                np.arange(image_y).reshape(1,-1,1,1) < end_ys.reshape(-1,1,1,1))
        mask = np.logical_and(mask_y, mask_x)
        mask = np.logical_and(mask, np.repeat(True, image_batch.shape[3]).reshape(1,1,1,-1))
        return mask

    def crop_concatenate(wk, hk, start_x, start_y, end_x, end_y):
        nonlocal output_images, output_labels
        xk = (np.random.rand(batch_size) * (image_x-wk)).astype(np.int32)
        yk = (np.random.rand(batch_size) * (image_y-hk)).astype(np.int32)
        target_indices = np.arange(batch_size)
        np.random.shuffle(target_indices)
        weights = wk * hk / image_x / image_y

        dest_mask = create_masks(start_x, start_y, end_x, end_y)
        target_mask = create_masks(xk, yk, xk+wk, yk+hk)

        output_images[dest_mask] = image_batch[target_indices][target_mask]
        output_labels += weights.reshape(-1, 1) * label_batch[target_indices]

    # left-top crop
    crop_concatenate(w, h, 
                     np.repeat(0, batch_size), np.repeat(0, batch_size), 
                     w, h)
    # right-top crop
    crop_concatenate(image_x-w, h, 
                     w, np.repeat(0, batch_size), 
                     np.repeat(image_x, batch_size), h)
    # left-bottom crop
    crop_concatenate(w, image_y-h, 
                     np.repeat(0, batch_size), h, 
                     w, np.repeat(image_y, batch_size))
    # right-bottom crop
    crop_concatenate(image_x-w, image_y-h, 
                     w, h, np.repeat(image_x, batch_size), 
                     np.repeat(image_y, batch_size))

    return output_images, output_labels

