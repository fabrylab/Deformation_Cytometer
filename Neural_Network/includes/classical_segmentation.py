
import numpy as np
from skimage.filters import gaussian
from scipy.ndimage import morphology
from skimage.measure import label, regionprops
import scipy
from skimage.transform import downscale_local_mean

def std_convoluted(image, N):
    im = np.array(image, dtype=float)
    im2 = im ** 2
    ones = np.ones(im.shape)
    kernel = np.ones((2 * N + 1, 2 * N + 1))
    if 1:
        kernel[0, 0] = 0
        kernel[-1, 0] = 0
        kernel[0, -1] = 0
        kernel[-1, -1] = 0
    s = scipy.signal.convolve2d(im, kernel, mode="same")
    s2 = scipy.signal.convolve2d(im2, kernel, mode="same")
    ns = scipy.signal.convolve2d(ones, kernel, mode="same")
    return np.sqrt((s2 - s ** 2 / ns) / ns)

def fill(im_sd, t=0.05):
    from skimage.morphology import flood
    im_sd[0, 0] = 0
    im_sd[0, -1] = 0
    im_sd[-1, 0] = 0
    im_sd[-1, -1] = 0
    mask = flood(im_sd, (0, 0), tolerance=t) | \
           flood(im_sd, (im_sd.shape[0] - 1, 0), tolerance=t) | \
           flood(im_sd, (0, im_sd.shape[1] - 1), tolerance=t) | \
           flood(im_sd, (im_sd.shape[0] - 1, im_sd.shape[1] - 1), tolerance=t)
    return mask


#config = getConfig(configfile)


class Segmentation():

    def __init__(self, pixel_size=None, r_min=None, frame_data=None, edge_dist=15, channel_width=0, **kwargs):
        # %% go through every frame and look for cells
        self.struct = morphology.generate_binary_structure(2, 1)  # structural element for binary erosion
        self.pixel_size = pixel_size

        #self.r_min = 5  # cells smaller than r_min (in um) will not be analyzed

        self.pixel_size = pixel_size * 1e6 # conversion to micrometer
        self.r_min = r_min
        self.frame_data = frame_data if frame_data is not frame_data else {}
        self.edge_dist = edge_dist
        self.config = {}
        self.config["channel_width_px"] = channel_width
        self.config["pixel_size_m"] = pixel_size

        self.Amin_pixels = np.pi * (self.r_min / self.pixel_size) ** 2  # minimum region area based on minimum radius
        self.down_scale_factor = 10
        self.edge_exclusion = 10  # in scale after downsampling
        self.count = 0
        self.success = 1

    def segmentation(self, img):
        cells = []
        img_cp = img.copy()
        if len(img.shape) == 3:
            img = img[:, :, 0]
        ellipses = []
        prediction_mask = np.zeros(img.shape)
        h, w = img.shape[0], img.shape[1]
        # flatfield correction
        img = img.astype(float) / np.median(img, axis=1)[:, None]
        #fig = plt.figure();plt.imshow(img),fig.savefig("/home/user/Desktop/out.png")
        im_high = scipy.ndimage.gaussian_laplace(img, sigma=1)  # kind of high-pass filtered image
        im_abs_high = np.abs(im_high)  # for detecting potential cells
        im_r = downscale_local_mean(im_abs_high, (self.down_scale_factor, self.down_scale_factor))
        im_rb = im_r > 0.010
        label_im_rb = label(im_rb)


        # region props are based on the downsampled abs high-pass image, row-column style (first y, then x)
        for region in regionprops(label_im_rb, im_r):

            if (region.max_intensity) > 0.03 and (region.area > self.Amin_pixels / 100):
                im_reg_b = label_im_rb == region.label
                min_row = region.bbox[0] * self.down_scale_factor - self.edge_exclusion
                min_col = region.bbox[1] * self.down_scale_factor - self.edge_exclusion
                max_row = region.bbox[2] * self.down_scale_factor + self.edge_exclusion
                max_col = region.bbox[3] * self.down_scale_factor + self.edge_exclusion

                if min_row > 0 and min_col > 0 and max_row < h and max_col < w:  # do not analyze cells near the edge
                    mask = fill(gaussian(im_abs_high[min_row:max_row, min_col:max_col], 3), 0.01)
                    mask = ~mask
                    mask = morphology.binary_erosion(mask, iterations=7).astype(int)
                    for subregion in regionprops(label(mask)):

                        if subregion.area > self.Amin_pixels:
                            ## Extract the mask
                            coords = subregion.coords ## consider min_row max_row....etc
                            prediction_mask[coords[:,0] + min_row, coords[:,1] + min_col] = 1
                            ## ellipses parameter
                            x_c = subregion.centroid[1] + min_col
                            y_c = subregion.centroid[0] + min_row

                          # this is to match clickpoints elipse angles checkout test2.py for illustration
                            angle = - subregion.orientation
                            if angle < 0:
                                angle = np.pi - np.abs(angle)
                            angle *= 180 / np.pi

                            a = subregion.major_axis_length / 2
                            b = subregion.minor_axis_length / 2
                            r = np.sqrt(a * b)

                            if  subregion.orientation > 0:
                                ellipse_angle = np.pi / 2 - subregion.orientation
                            else:
                                ellipse_angle = -np.pi / 2 - subregion.orientation

                            # the circumference of the ellipse
                            circum = np.pi * ((3 * (a + b)) - np.sqrt(10 * a * b + 3 * (a ** 2 + b ** 2)))

                            # %% compute radial intensity profile around each ellipse
                            theta = np.arange(0, 2 * np.pi, np.pi / 8)

                            i_r = np.zeros(int(3 * r))
                            for d in range(0, int(3 * r)):
                                # get points on the circumference of the ellipse
                                x = d / r * a * np.cos(theta)
                                y = d / r * b * np.sin(theta)
                                # rotate the points by the angle fo the ellipse
                                t = ellipse_angle
                                xrot = (x * np.cos(t) - y * np.sin(t) +  x_c).astype(int)
                                yrot = (x * np.sin(t) + y * np.cos(t) + y_c).astype(int)
                                # crop for points inside the iamge
                                index = (xrot < 0) | (xrot >= img.shape[1]) | (yrot < 0) | (yrot >= img.shape[0])
                                x = xrot[~index]
                                y = yrot[~index]
                                # average over all these points
                                i_r[d] = np.mean(img[y, x])

                            # define a sharpness value
                            sharp = (i_r[int(r + 2)] - i_r[int(r - 2)]) / 5 / np.std(i_r)

                            # %% store the cells
                            yy = subregion.centroid[0] - self.config["channel_width_px"] / 2
                            yy = yy * self.config["pixel_size_m"] * 1e6

                            data = {}
                            data.update(self.frame_data)
                            data.update({
                                "x_pos": x_c,  # x_pos
                                "y_pos": y_c,  # y_pos
                                "radial_pos": yy,  # RadialPos
                                "long_axis": float(format(subregion.major_axis_length)) * self.config["pixel_size_m"] * 1e6,
                                # LongAxis
                                "short_axis": float(format(subregion.minor_axis_length)) * self.config[
                                    "pixel_size_m"] * 1e6,  # ShortAxis
                                "angle": np.rad2deg(ellipse_angle),  # angle
                                "irregularity": subregion.perimeter / circum,  # irregularity
                                "solidity": subregion.solidity,  # solidity
                                "sharpness": sharp,  # sharpness
                            })
                            cells.append(data)

        return prediction_mask, cells



