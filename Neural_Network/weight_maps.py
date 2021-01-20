import numpy as np
from skimage.morphology import label
from scipy.ndimage.morphology import distance_transform_edt
from skimage.morphology import binary_erosion, binary_dilation
from tqdm import tqdm
from skimage.morphology import remove_small_holes


# unet like weighting --> explicitly for cell cell interfaces:
def weight_map_unet(mask, class_weight=1, sigma=1, w0=10):
    y, x = np.indices(mask.shape)
    mask_l = label(mask)
    dists = np.array([distance_transform_edt(~(mask_l == l)) for l in np.unique(mask_l)[1:]])
    min_dists_coord = np.argmin(dists, axis=0)
    min_dist = dists[min_dists_coord.flatten(), y.flatten(), x.flatten()].reshape(mask.shape)
    dists[min_dists_coord.flatten(), y.flatten(), x.flatten()] = np.inf
    sec_min_dists_coord = np.argmin(dists, axis=0)
    sec_min_dist = dists[sec_min_dists_coord.flatten(), y.flatten(), x.flatten()].reshape(mask.shape)

    weight = class_weight + w0 * np.exp(-(min_dist + sec_min_dist) ** 2 / (2 * sigma ** 2))
    weight[mask.astype(bool)] = class_weight
    return weight


def generate_weight_map(masks, function=None, **kwargs):

    if len(masks.shape) == 2:
        return(function(masks, **kwargs))

    print("generating weight maps")
    ws = []
    for m in tqdm(masks):
        ws.append(function(m[:, :, 0], **kwargs))
    return np.array(ws)


def weight_background(mask, ws, **kwargs):

    weights = np.zeros(mask.shape, dtype=np.float32)
    weights[mask.astype(bool)] =  ws[0]
    weights[~mask.astype(bool)] = ws[1]
    return weights



def extract_edge(y, thickness=3):
   
    if len(y.shape) == 2:
        inner = y.copy()
        for i in range(thickness):
            inner = binary_erosion(inner)
        out = (~inner & y).astype(int)
        out[inner] = 0
    else:
        out = []
        for ma in tqdm(y):
            m = ma[:,:,0] > 0
            inner = m.copy()
            for i in range(thickness):
                inner = binary_erosion(inner)
            o = (~inner & m).astype(int)
            o[inner] = 0
            out.append(o[:,:,None])
    return np.array(out).astype(int)


