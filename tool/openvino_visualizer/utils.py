from scipy.ndimage import gaussian_filter, maximum_filter
import cv2
import numpy as np
BODY_PARTS_KPT_IDS = [[15, 13],
                      [13, 11],
                      [16, 14],
                      [14, 12],
                      [5, 11],
                      [6, 12],
                      [5, 7],
                      [6, 8],
                      [7, 9],
                      [8, 10],
                      [0, 1],
                      [0, 2],
                      [1, 3],
                      [2, 4],
                      [0, 5],
                      [0, 6],
                      [0, 1],
                      [0, 2]]

# BODY_PARTS_KPT_IDS = [[15, 13],
#                       [13, 11],
#                       [16, 14],
#                       [14, 12],
#                       [5, 11],
#                       [6, 12],
#                       [5, 7],
#                       [6, 8],
#                       [7, 9],
#                       [8, 10]]


def post_process_heatmap(heatMap):
    kplst = list()
    for i in range(heatMap.shape[0]):
        _map = heatMap[i, :, :]
        _map = gaussian_filter(_map, sigma=1)
        _nmsPeaks = non_max_supression(_map, windowSize=3, threshold=1e-6)

        y, x = np.where(_nmsPeaks == _nmsPeaks.max())
        if len(x) > 0 and len(y) > 0:
            kplst.append((int(x[0]), int(y[0]), _nmsPeaks[y[0], x[0]]))
        else:
            kplst.append((0, 0, 0))
    kp = np.array(kplst)
    return kp


def non_max_supression(plain, windowSize=3, threshold=1e-6):
    # clear value less than threshold
    under_th_indices = plain < threshold
    plain[under_th_indices] = 0
    return plain * (plain == maximum_filter(plain, footprint=np.ones((windowSize, windowSize))))


def render_kps(cvmat, kps, scale_x, scale_y, thr=0.1):
    for _kp in kps:
        _x, _y, _conf = _kp
        if _conf > thr:
            cv2.circle(cvmat, center=(int(_x*4*scale_x), int(_y*4*scale_y)), color=(0, 0, 255), radius=5)
    return cvmat


def visualize(image, kps, scale_x, scale_y, thr=0.1):
    num_kpts = 17
    assert kps.shape[0] == num_kpts
    for part_id in range(len(BODY_PARTS_KPT_IDS)):
        kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
        kps_a_conf = kps[kpt_a_id, 2]
        kpa = kps[kpt_a_id, :2]
        if kps_a_conf > thr:
            x_a, y_a = int(kpa[0] * 4 * scale_x), int(kpa[1] * 4 * scale_y)
            cv2.circle(image, center=(x_a, y_a), color=(0, 0, 255), radius=2)
        kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
        kps_b_conf = kps[kpt_b_id, 2]
        kpb = kps[kpt_b_id, :2]
        if kps_b_conf > thr:
            x_b, y_b = int(kpb[0] * 4 * scale_x), int(kpb[1] * 4 * scale_y)
            cv2.circle(image, center=(x_b, y_b), color=(0, 0, 255), radius=2)
        if kps_a_conf > thr and kps_b_conf > thr:
            cv2.line(image, (x_a, y_a), (x_b, y_b), (0, 255, 255), 2)


