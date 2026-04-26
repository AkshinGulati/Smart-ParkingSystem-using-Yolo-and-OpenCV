import numpy as np
from filterpy.kalman import KalmanFilter

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))

def iou_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w, h = np.maximum(0., xx2 - xx1), np.maximum(0., yy2 - yy1)
    inter = w * h
    return inter / ((bb_test[...,2]-bb_test[...,0])*(bb_test[...,3]-bb_test[...,1]) +
                    (bb_gt[...,2]-bb_gt[...,0])*(bb_gt[...,3]-bb_gt[...,1]) - inter)

def convert_bbox_to_z(b):
    w, h = b[2]-b[0], b[3]-b[1]
    return np.array([b[0]+w/2, b[1]+h/2, w*h, w/float(h)]).reshape((4,1))

def convert_x_to_bbox(x):
    w = np.sqrt(x[2]*x[3])
    h = x[2]/w
    return np.array([x[0]-w/2, x[1]-h/2, x[0]+w/2, x[1]+h/2]).reshape((1,4))

class KalmanBoxTracker:
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],[0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.P[4:,4:] *= 1000
        self.kf.P *= 10
        self.kf.R[2:,2:] *= 10
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.id = KalmanBoxTracker.count; KalmanBoxTracker.count += 1
        self.time_since_update, self.hit_streak = 0, 0

    def update(self, bbox):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0: self.kf.x[6] = 0
        self.kf.predict()
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x)

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)

def associate(dets, trks, iou_th=0.3):
    if len(trks)==0:
        return np.empty((0,2),int), np.arange(len(dets)), np.empty((0,),int)

    iou = iou_batch(dets, trks)
    matched = linear_assignment(-iou)

    unmatched_dets = [d for d in range(len(dets)) if d not in matched[:,0]]
    unmatched_trks = [t for t in range(len(trks)) if t not in matched[:,1]]

    matches = []
    for m in matched:
        if iou[m[0], m[1]] < iou_th:
            unmatched_dets.append(m[0])
            unmatched_trks.append(m[1])
        else:
            matches.append(m)
    return np.array(matches), np.array(unmatched_dets), np.array(unmatched_trks)

class Sort:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        self.max_age, self.min_hits, self.iou_th = max_age, min_hits, iou_threshold
        self.trackers, self.frame_count = [], 0

    def update(self, dets=np.empty((0,5))):
        self.frame_count += 1
        trks = np.array([t.predict()[0] for t in self.trackers]) if self.trackers else np.empty((0,4))

        matched, ud, ut = associate(dets, trks, self.iou_th)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])

        for i in ud:
            self.trackers.append(KalmanBoxTracker(dets[i]))

        ret = []
        for t in reversed(self.trackers):
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.append(t.get_state()[0], t.id+1))
            if t.time_since_update > self.max_age:
                self.trackers.remove(t)

        return np.array(ret) if ret else np.empty((0,5))