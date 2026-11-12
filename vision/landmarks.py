"""Two-hand features. Operate on unmirrored RGB inputs, never mirrored preview pixels."""
from __future__ import annotations
import numpy as np

FEATURE_VERSION = "two-hand-shared-xy-v1"
FEATURE_COUNT = 128
CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

def validate_arrays(points, mask):
    points = np.asarray(points, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.uint8)
    if points.shape != (2,21,3) or mask.shape != (2,):
        raise ValueError("Expected points (2,21,3) and mask (2,)")
    if not np.isfinite(points).all() or not np.isin(mask,[0,1]).all():
        raise ValueError("Nonfinite landmarks or invalid mask")
    return points, mask

def normalize(points, mask, width, height):
    """128 floats: left/right 63 coordinates plus two explicit presence flags.

    x and z are aspect-corrected to image-height units. A common wrist origin
    and palm scale preserve BETWEEN-hand x/y displacement. Each hand's z is
    wrist-relative, because MediaPipe image z does not share inter-hand depth.
    Rotation is deliberately preserved, since orientation can distinguish signs.
    Missing hands remain zero with mask=0. No hands returns an all-zero vector.
    """
    points,mask=validate_arrays(points,mask)
    if width<=0 or height<=0:raise ValueError("Image dimensions must be positive")
    if not mask.any():return np.zeros(FEATURE_COUNT,dtype=np.float32)
    data=points.copy();data[:,:,0]*=width/height;data[:,:,2]*=width/height
    present=np.flatnonzero(mask)
    origin=data[present[0],0,:2].copy()
    scales=[np.linalg.norm(data[i,9,:2]-data[i,0,:2]) for i in present]
    scale=float(np.median(scales))
    if scale<1e-5:raise ValueError("Degenerate palm scale")
    result=np.zeros_like(data)
    for i in present:
        result[i,:,:2]=(data[i,:,:2]-origin)/scale
        result[i,:,2]=(data[i,:,2]-data[i,0,2])/scale
    return np.concatenate([result.reshape(-1),mask.astype(np.float32)])

def unpack_result(result):
    """Use handedness slots instead of sorting hands anew on every frame.

    Low-confidence or duplicate handedness assignments are excluded rather
    than silently swapping slots. Strong occlusion can still break tracking.
    """
    points=np.zeros((2,21,3),dtype=np.float32);mask=np.zeros(2,dtype=np.uint8)
    scores=np.zeros(2,dtype=np.float32)
    for landmarks,handedness in zip(result.hand_landmarks,result.handedness):
        if not handedness:continue
        side=handedness[0]
        if side.category_name not in ("Left","Right") or side.score<0.5:continue
        slot=0 if side.category_name=="Left" else 1
        if side.score<=scores[slot]:continue
        points[slot]=np.array([[p.x,p.y,p.z] for p in landmarks],dtype=np.float32)
        mask[slot]=1;scores[slot]=side.score
    return points,mask

class Detector:
    def __init__(self,asset_path,video=True,min_detection_confidence=.5,min_tracking_confidence=.5):
        import mediapipe as mp
        self.mp=mp
        options=mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(asset_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO if video else mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,min_hand_detection_confidence=min_detection_confidence,min_tracking_confidence=min_tracking_confidence)
        self.engine=mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.video=video;self.last_timestamp=-1
    def detect(self,rgb,timestamp_ms=0):
        image=self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=np.ascontiguousarray(rgb))
        if self.video:
            timestamp_ms=max(int(timestamp_ms),self.last_timestamp+1);self.last_timestamp=timestamp_ms
            result=self.engine.detect_for_video(image,timestamp_ms)
        else:result=self.engine.detect(image)
        return unpack_result(result)
    def close(self):self.engine.close()
