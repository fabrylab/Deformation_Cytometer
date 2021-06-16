from tqdm import tqdm
from deformationcytometer.detection.includes.regionprops import mask_to_cells, mask_to_cells_edge
from scipy.optimize import linear_sum_assignment
from Neural_Network.includes.classical_segmentation import Segmentation as Segmentation_classical
from Neural_Network.includes.data_handling import *
from Neural_Network.includes.data_handling import *
from deformationcytometer.detection.includes.UNETmodel import UNet
import clickpoints
import pandas as pd

pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 500)
pd.set_option('display.max_colwidth', 20)


class Segmentation:
    # basic segmentation class that can
    # 1) store an network
    # 2) handle both segmentation based on cell area and cell boundary
    # 3) return the segmentation mask

    def __init__(self, network_path=None, img_shape=None, pixel_size=None, r_min=None, frame_data=None, edge_dist=15,
                 channel_width=0, edge_only=False, return_mask=True, d=8, **kwargs):

        self.unet = UNet((img_shape[0], img_shape[1], 1), 1, d=d)
        self.unet.load_weights(network_path)
        self.pixel_size = pixel_size
        self.r_min = r_min
        self.frame_data = frame_data if frame_data is not frame_data else {}
        self.edge_dist = edge_dist
        self.config = {}
        self.config["channel_width_px"] = channel_width
        self.config["pixel_size_m"] = pixel_size
        self.edge_only = edge_only
        self.return_mask = return_mask

    def search_cells(self, prediction_mask, img):

        if self.edge_only:
            if self.return_mask:
                cells, prediction_mask = mask_to_cells_edge(prediction_mask, img, self.config, self.r_min,
                                                            self.frame_data, self.edge_dist,
                                                            return_mask=self.return_mask)
            else:
                cells = mask_to_cells_edge(prediction_mask, img, self.config, self.r_min, self.frame_data,
                                           self.edge_dist, return_mask=self.return_mask)
        else:
            cells = mask_to_cells(prediction_mask, img, self.config, self.r_min, self.frame_data, self.edge_dist)

        return prediction_mask, cells

    def segmentation(self, img):
        # image batch
        if len(img.shape) == 4:
            img = preprocess_batch(img)
            prediction_mask = self.unet.predict(img) > 0.5
            cells = []
            for i in range(prediction_mask.shape[0]):
                _, cells_ = self.search_cells(prediction_mask[i, :, :, 0], img[i, :, :, 0])
                cells.extend(cells_)
            prediction_mask = None
        # single image
        elif len(img.shape) == 2:

            img = (img - np.mean(img)) / np.std(img).astype(np.float32)
            prediction_mask = self.unet.predict(img[None, :, :, None])[0, :, :, 0] > 0.5
            prediction_mask, cells = self.search_cells(prediction_mask, img)
        else:
            raise Exception("incorrect image shape: img.shape == " + str(img.shape))
        return prediction_mask, cells


def load_ellipses(cdb, img_o):
    # Deprecated (?)
    # loading ellipses from ClickPoints database
    ellipses_o = np.array([[e.x, e.y, e.width, e.height, e.angle] for e in cdb.getEllipses(image=img_o)])
    if ellipses_o.size > 0:
        for i in range(np.shape(ellipses_o)[0]):
            if ellipses_o[i, 2] < ellipses_o[i, 3]:
                ellipses_o[i, 2] = ellipses_o[i, 3]
                ellipses_o[i, 3] = ellipses_o[i, 2]
                ellipses_o[i, 4] = ellipses_o[i, 4] - 90
    return ellipses_o


def load_polygones(cdb, img_o, ny, nx, points, im, config, r_min, frame_data, edge_dist, irr_th=1.06, sol_th=0.96):
    # Loading ellipses/cells from ClickPoints polygon markers.
    q_polys = cdb.getPolygons(image=img_o)
    mask = np.zeros(img_o.getShape()[0:2], dtype=np.uint8)
    for pol in q_polys:
        polygon = np.array([pol.points])
        if len(polygon[0]) > 5:  # Ground Truth may contain accidentally clicked small polygons
            path = Path(polygon.squeeze())
            grid = path.contains_points(points)
            grid = grid.reshape((ny, nx))
            mask += grid
    cells = mask_to_cells(mask, im, config, r_min, frame_data, edge_dist)
    cells_filtered = filter_cells(cells, irr_th, sol_th)
    ellipses = np.array(
        [[cell["y"], cell["x"], cell["long_axis"], cell["short_axis"], cell["angle"], getStrain(cell)] for cell
         in cells])
    ellipses_filtered = np.array(
        [[cell["y"], cell["x"], cell["long_axis"], cell["short_axis"], cell["angle"], getStrain(cell)] for cell
         in cells_filtered])
    return mask, ellipses, ellipses_filtered, q_polys


def filter_cells(cells, irregularity_th=1.06, solidity_th=0.96):
    # filter cells based on irregularity and solidity
    return [cell for cell in cells if cell["irregularity"] < irregularity_th and cell["solidity"] > solidity_th]


def match_to_gt(elipses_gt, elipses_pred):
    # matching between predcion and ground truth bases on linear sum assignment
    # --> first the prediction and gt are matched based on a cost function (euclidean distance
    # of ellipse centers) os that the sum of all distances is minimized. Then pairings with distance greater then 50
    # pixels are removed. This is not equivalent to nearest neighbour matching!!
    if elipses_gt.size == 0 or elipses_pred.size == 0:
        return [], [], [], []
    cost = np.linalg.norm(elipses_gt[None, :, :2] - elipses_pred[:, None, :2], axis=-1)
    sys_ids, gt_ids = linear_sum_assignment(cost)

    gt_texts = np.array(["%d" % i for i in range(len(elipses_gt))])
    sys_texts = np.array(["x" for i in range(len(elipses_pred))])
    sys_texts[sys_ids] = np.array(["%d" % i for i in gt_ids])
    index = []
    for i in range(len(sys_ids)):
        if cost[sys_ids[i], gt_ids[i]] > 50:
            index.append(i)
    sys_ids = np.delete(sys_ids, index)
    gt_ids = np.delete(gt_ids, index)

    return sys_ids, gt_ids, sys_texts, gt_texts


def img_ids_until_first_none(cdb):
    # Maybe unnecessary?
    ids = []
    i = 1
    imgs = cdb.getImage(id=i)
    while imgs is not None:
        ids.append(i)
        i += 1
        imgs = cdb.getImage(id=i)
    return ids


def collect_evaluation(ellipses, ellipses_gt, regularity_, store_list):
    # writing the pairing results to a number of lists
    GT, GTMatch, SysMatch, id_Sys, Sys_new, Sys, regularity = store_list
    regularity.extend(regularity_)
    GT.extend(ellipses_gt)
    Sys.extend(ellipses)
    if ellipses_gt.size > 0:
        # pair gt with prediction 1
        sys_ids, gt_ids, sys_texts, gt_texts = match_to_gt(ellipses_gt, ellipses)
        GTMatch.extend(ellipses_gt[gt_ids])
        SysMatch.extend(ellipses[sys_ids])
        id_Sys.append(id)
    else:
        Sys_new.extend(ellipses)


def set_some_parameters(parameters_dict):

    # pixel_size_camera, channel_width, r_min and edge_dist are given in µm
    magnification = parameters_dict["magnification"]
    coupler = parameters_dict["coupler"]
    pixel_size_camera = parameters_dict["pixel_size_camera"]
    pixel_size = pixel_size_camera / (magnification * coupler)
    pixel_size *= 1e-6  # conversion to meter
    r_min = parameters_dict["r_min"]  # in µm
    edge_dist = parameters_dict["edge_dist"]  # in µm
    channel_width = parameters_dict["channel_width"]
    config = {}
    config["channel_width_px"] = channel_width
    config["pixel_size_m"] = pixel_size
    frame_data = {}

    return magnification, coupler, pixel_size, r_min, edge_dist, channel_width, config, frame_data


def getStrain(cell):

    return (cell["long_axis"] - cell["short_axis"]) / np.sqrt(cell["long_axis"] * cell["short_axis"])


def evaluate_database(result, result_filtered, cdb_file, network_path, parameters_dict, network_names, edge_only=False,
                      d=8, irr_th=1.06, sol_th=0.96, batch_size=10):
    """
    main function that evaluates a neural network on a ClickPoints database containing the
    ground truth for cell detection.

    @param result: pandas data frame to with the evaluation results will be appended as one row
    @param result_filtered:  pandas data frame to with the evaluation results without applying regularity or solidity
            thresholds will be appended as one row
    @param cdb_file: full path to a ClickPoints database
    @param network_path: full path to a neural network weight file. The network must be based on the Unet defined in
            deformationcytometer.detection.inculdes.UNETmodel.py
    @param parameters_dict: dictionary with additional measurement parameters (see evaluation.py for example)
    @param network_names: name of the neural network that will be used in the data frames and in the plots.
                Can be anything.
    @param edge_only: use the cell detection pipeline based on segmentation of the cell boundary only. The main feature
            of this pipeline is that any object, that does not fully enclose an area (the cell center) is removed
    @param d: Number of filters in the first layer of the neural network.
    @param irr_th: Threshold for irregularity. We recommend to use the "Deformation Cytometer" addon for ClickPoints
            to optimize these thresholds.
    @param sol_th: Threshold for solidity. We recommend to use the "Deformation Cytometer" addon for ClickPoints
            to optimize these thresholds.
    @param batch_size: Batch size of the Neural Network cell detection. Large batch sizes are faster but may exceed your
            hardware capabilities (ResourceExhaustedError). Try batch_size = 1 if you run into such problems
    """


    magnification, coupler, pixel_size, r_min, channel_width, edge_dist, config, frame_data = set_some_parameters(
        parameters_dict)
    GT, GTMatch, SysMatch, id_Sys, Sys_new, Sys, regularity = [], [], [], [], [], [], []
    GTf, GTMatchf, SysMatchf, id_Sysf, Sys_newf, Sysf, regularityf = [], [], [], [], [], [], []
    cdb = clickpoints.DataFile(cdb_file)
    # get typical images shape
    img_o = cdb.getImages()[0]
    im_shape = list(img_o.data.shape)
    # build the neural network
    if network_path.endswith(".h5"):
        Seg = Segmentation(network_path=network_path,
                           img_shape=im_shape, pixel_size=pixel_size, r_min=r_min, frame_data={},
                           edge_dist=edge_dist, channel_width=channel_width, edge_only=edge_only, return_mask=False,
                           d=d)
    elif network_path == "classical":
        Seg = Segmentation_classical(network_path=network_path,
                                     img_shape=im_shape, pixel_size=pixel_size, r_min=r_min, frame_data={},
                                     edge_dist=edge_dist, channel_width=channel_width, edge_only=edge_only,
                                     return_mask=False)
    else:
        raise Exception("did not recognize " + network_path)
    # get indices array for matplotlib Path ellipse fit
    ny, nx = im_shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T
    # print number of ellipses or polygones found in the current database
    q_elli_count = cdb.getEllipses().count()
    polys_count = cdb.getPolygons().count()
    print("%d ellipses found" % q_elli_count, "%d polygons found" % polys_count)

    # all image ids until the first image entry is None or until file_id_range
    ids = img_ids_until_first_none(cdb)

    # splitting into batches
    id_batches = np.array_split(np.array(ids), len(ids) / batch_size)
    # iterating through all images
    for id_b in tqdm(id_batches):
        ims = []
        ellipses_gt = []
        ellipses_filtered_gt = []
        for id in id_b:
            # load image
            img_gt = cdb.getImage(id=id)
            img = img_gt.get_data()
            if len(img.shape) == 3:
                img = img[:, :, 0]
            ims.append(img)
            # get ground truth ellipses in this image
            # either by loading ellipse marker directly or by fitting to polygon
            if q_elli_count > 0:
                el_gt = load_ellipses(cdb, img_gt)
                ellipses_gt.extend(el_gt)
            if polys_count > 0:
                mask, el_gt, el_filtered_gt, q_polys = load_polygones(cdb, img_gt, ny, nx,
                                                                      points, img, config, r_min,
                                                                      frame_data, edge_dist, irr_th, sol_th)
                ellipses_gt.extend(el_gt)
                ellipses_filtered_gt.extend(el_filtered_gt)
        ellipses_gt = np.array(ellipses_gt)
        ellipses_filtered_gt = np.array(ellipses_filtered_gt)
        ims = np.array(ims)

        # prediction
        prediction_mask, cells = Seg.segmentation(ims[:, :, :, None])
        # apply regularity and solidity filters to prediction
        cells_filtered = filter_cells(cells, irr_th, sol_th)
        ellipses = np.array(
            [[cell["y"], cell["x"], cell["long_axis"], cell["short_axis"], cell["angle"], getStrain(cell)] for
             cell in cells])
        ellipses_filtered = np.array(
            [[cell["y"], cell["x"], cell["long_axis"], cell["short_axis"], cell["angle"], getStrain(cell)] for
             cell in cells_filtered])
        regus = [cell["irregularity"] for cell in cells]
        regus_filtered = [cell["irregularity"] for cell in cells_filtered]

        # perform matching to ground truth and remember ground truth and prediction
        collect_evaluation(ellipses, ellipses_gt, regus, [GT, GTMatch, SysMatch, id_Sys, Sys_new, Sys, regularity])
        collect_evaluation(ellipses_filtered, ellipses_filtered_gt, regus_filtered, [GTf, GTMatchf, SysMatchf, id_Sysf,
                                                                                     Sys_newf, Sysf, regularityf])
    cdb.db.close()  # close database
    GT, Sys, GTMatch, SysMatch, Sys_new, regularity = convert_arrays([GT, Sys, GTMatch, SysMatch, Sys_new, regularity])
    GTf, Sysf, GTMatchf, SysMatchf, Sys_newf, regularityf = convert_arrays(
        [GTf, Sysf, GTMatchf, SysMatchf, Sys_newf, regularityf])
    result.loc[len(result)] = [cdb_file, network_names[network_path], network_path, GT, Sys, GTMatch, SysMatch, Sys_new,
                               regularity]
    result_filtered.loc[len(result_filtered)] = [cdb_file, network_names[network_path], network_path, GTf, Sysf,
                                                 GTMatchf,
                                                 SysMatchf, Sys_newf, regularityf]


def convert_arrays(arrs):

    return [np.array(ar) for ar in arrs]


def get_recall_precision(GT, SysMatch, Sys):

    TP = np.shape(SysMatch)[0]
    FN = np.shape(GT)[0] - TP
    FP = np.shape(Sys)[0] - TP
    if len(Sys) == 0:
        precision = np.nan
    else:
        precision = TP / (TP + FP)
    if len(GT) == 0:
        recall = np.nan
    else:
        recall = TP / (TP + FN)
    F1_score = (2 * precision * recall) / (precision + recall)
    return precision, recall, F1_score


def get_differences_median(match1, match2):

    return np.median(match1, axis=0) - np.median(match2, axis=0)


def get_differences_error(match1, match2, squared=False):
    # calculating the mean absolute error for cell x-, y-position, width, height, orientation angle and strain
    # between ground truth and prediction. Only for matched cells
    if len(match1) == 0 or len(match2) == 0:
        return [np.array([]) for i in range(6)]

    x_pos_diff = match1[:, 0] - match2[:, 0]
    y_pos_diff = match1[:, 1] - match2[:, 1]
    width_diff = match1[:, 2] - match2[:, 2]
    height_diff = match1[:, 3] - match2[:, 3]
    angle_diff = (match2[:, 4] + 90) % 180 - 90 - (
            (match1[:, 4] - 90 * (match1[:, 2] < match1[:, 3]) + 90) % 360 - 90)
    strain_diff = (match1[:, 5] - match2[:, 5])
    values = [x_pos_diff, y_pos_diff, width_diff, height_diff, angle_diff, strain_diff]
    if squared:
        values = [v ** 2 for v in values]

    return values


def add_measures(data_frame):
    # adding the mean errors and recall, prediction and F1_score to the data frame
    types = ["x_diff", "y_diff", "w_diff", "h_diff", "a_diff", "strain_diff"]
    # add errors
    diffs = []
    for i, row in data_frame.iterrows():
        out_dict = {t: v for t, v in zip(types, get_differences_error(row["GT Match"], row["Pred Match"]))}
        out_dict_mean = {t + "_mean": np.mean(v) for t, v in
                         zip(types, get_differences_error(row["GT Match"], row["Pred Match"]))}
        out_dict_med = {t + "_med": v for t, v in zip(types, get_differences_median(row["GT"], row["Pred"]))}
        precision, recall, F1_score = get_recall_precision(row["GT"], row["Pred Match"], row["Pred"])
        if np.isnan(F1_score):
            print("####")
        out_dict.update(out_dict_med)
        out_dict.update(out_dict_mean)
        out_dict.update({"precision": precision, "recall": recall, "F1_score": F1_score})
        diffs.append(out_dict)

    diffs = pd.DataFrame(diffs)
    for key in diffs.keys():
        data_frame[key] = diffs[key]


def write_result_summary(res, out_folder, name):
    # writing a easily readable csv file
    res_ = res.copy()
    types = ["network", "x_diff_mean", "y_diff_mean", "w_diff_mean", "h_diff_mean",
             "a_diff_mean", "strain_diff_mean", "total detections", "precision", "recall", "F1_score", "network_path",
             "database"]
    res_ = res_[types]
    res_.to_csv(os.path.join(out_folder, name))
