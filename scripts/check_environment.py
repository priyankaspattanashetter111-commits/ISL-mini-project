"""Verify imports and a REAL MediaPipe inference on a blank image. No camera opened."""
from pathlib import Path
import json,platform,sys,importlib.metadata,subprocess,shutil

def main():
    status={'python':sys.version.split()[0],'platform':platform.platform(),'packages':{},'checks':{},'camera_tested':False,'training_performed':False}
    if sys.version_info[:2]!=(3,12):raise SystemExit('Use Python 3.12 for this pinned Windows environment.')
    for package in ['mediapipe','opencv-contrib-python','numpy','pytest']:
        try:status['packages'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:raise SystemExit('Missing '+package+'; install requirements.txt')
    import cv2,numpy as np
    from vision.landmarks import Detector,normalize
    path=Path('models/hand_landmarker.task')
    if not path.exists():raise SystemExit('Run python -m scripts.download_detector first.')
    detector=Detector(path,video=False)
    try:
        points,masks=detector.detect(np.zeros((240,320,3),dtype=np.uint8))
        assert masks.sum()==0, 'Blank frame unexpectedly detected a hand'
        assert normalize(points,masks,320,240).shape==(128,)
    finally:detector.close()
    status['checks']={'opencv_import':True,'mediapipe_detector_created':True,'blank_frame_inference':True,'no_hand_mask':True,'feature_shape_128':True}
    status['node_available']=shutil.which('node') is not None
    status['mongodb_available']=shutil.which('mongod') is not None
    Path('reports').mkdir(exist_ok=True)
    Path('reports/environment.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2));print('PASS: environment and detector checks. Camera and signer collection still need a local test.')
if __name__=='__main__':main()
