"""OpenCV collector. Save one independent landmark clip per deliberate performance."""
import argparse,json,time,sys
from pathlib import Path
import cv2,numpy as np
from vision.landmarks import Detector,normalize,CONNECTIONS
from vision.storage import clip_quality,save_clip,safe_id

def open_camera(index):
    cap=cv2.VideoCapture(index,cv2.CAP_DSHOW) if sys.platform=='win32' else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release();cap=cv2.VideoCapture(index)
    if not cap.isOpened():raise RuntimeError('Camera unavailable. Close Zoom/Teams and allow desktop camera access in Windows Settings. Try --camera 1.')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,960);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    return cap

def draw_preview(frame,points,masks,width=960):
    h,w=frame.shape[:2];canvas=frame.copy()
    for slot in np.flatnonzero(masks):
        color=(178,220,52) if slot==0 else (255,171,76)
        pixels=[(int(x*w),int(y*h)) for x,y,z in points[slot]]
        for a,b in CONNECTIONS:cv2.line(canvas,pixels[a],pixels[b],color,2)
        for point in pixels:cv2.circle(canvas,point,4,color,-1)
    canvas=cv2.flip(canvas,1) # display only; inference and saved coordinates remain unmirrored
    scale=width/w;return cv2.resize(canvas,(width,int(h*scale)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preview',action='store_true',help='Camera/hand check only; saves no data')
    p.add_argument('--label',choices=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    p.add_argument('--signer',default='');p.add_argument('--session',default='')
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--lighting',choices=['normal','dim','bright'],default='normal')
    p.add_argument('--angle',choices=['front','left','right','tilted'],default='front')
    p.add_argument('--distance',choices=['near','medium','far'],default='medium')
    p.add_argument('--consent',action='store_true',help='Signer agreed to this local landmark collection')
    a=p.parse_args();settings=json.loads(Path('config/capture.json').read_text());mapping=json.loads(Path('config/alphabet.json').read_text())
    spec=None
    if not a.preview:
        if not a.label or not a.signer or not a.session or not a.consent:p.error('Recording requires --label, --signer, --session and --consent. Use --preview for no-save camera checks.')
        safe_id(a.signer);safe_id(a.session);spec=mapping['labels'][a.label]
        if not spec['reviewed']:raise SystemExit('Mapping unreviewed. Run scripts.review_label only after a human review. Use --preview to test the camera now.')
    asset=Path('models/hand_landmarker.task')
    if not asset.exists():raise SystemExit('First run: python -m scripts.download_detector')
    cap=open_camera(a.camera)
    detector=None;recording=False;arrays=None;clip_start=0.;countdown=None;saved=0
    note='Preview only: no data saved' if a.preview else 'SPACE: countdown, then perform ONE sign. Q: quit.'
    title='ISL Dataset Studio - Phase 1'
    try:
        detector=Detector(asset,video=True,min_detection_confidence=settings['minimum_detection_confidence'],min_tracking_confidence=settings['minimum_tracking_confidence'])
        while True:
            loop=time.monotonic();ok,frame=cap.read()
            if not ok:raise RuntimeError('Camera frame failed. Check permissions and camera availability.')
            h,w=frame.shape[:2];ts=int(time.monotonic()*1000)
            points,masks=detector.detect(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),ts)
            try:features=normalize(points,masks,w,h)
            except ValueError:points=np.zeros_like(points);masks=np.zeros_like(masks);features=np.zeros(128,dtype=np.float32)
            now=time.monotonic()
            if countdown is not None and now>=countdown:
                recording=True;countdown=None;clip_start=now
                arrays={'points':[],'masks':[],'features':[],'timestamps':[],'width':w,'height':h}
            if recording:
                arrays['points'].append(points.copy());arrays['masks'].append(masks.copy());arrays['features'].append(features.copy());arrays['timestamps'].append(ts)
                if now-clip_start>=settings['clip_seconds']:
                    quality=clip_quality(arrays['masks'],spec['expected_hands'],settings['minimum_frames'],settings['minimum_valid_ratio'])
                    if quality['accepted']:
                        metadata={'mapping_version':mapping['mapping_version'],'mapping_reviewed':True,'gesture_kind':spec['kind'],'expected_hands':spec['expected_hands'],'reference':spec['reference_url'],'reviewer_id':spec['reviewer_id'],'consent_confirmed':True,'lighting':a.lighting,'angle':a.angle,'distance':a.distance,'image_width':w,'image_height':h,'quality':quality,'duration_seconds':round(now-clip_start,3)}
                        path=save_clip('data/raw',a.signer,a.session,a.label,arrays['points'],arrays['masks'],arrays['features'],arrays['timestamps'],metadata)
                        saved+=1;note=f'Saved clip {saved}. Relax hands before the next performance.';print('SAVED',path,quality,flush=True)
                    else:note=f"Rejected: {quality['valid_ratio']:.0%} valid frames. Keep required hands visible.";print('REJECTED',quality,flush=True)
                    recording=False;arrays=None
            image=draw_preview(frame,points,masks,settings['preview_width'])
            overlay=np.zeros((170,image.shape[1],3),dtype=np.uint8);overlay[:]=(30,34,24)
            label='Camera readiness' if a.preview else f"{a.label} | {a.signer}/{a.session} | {spec['kind']}"
            if countdown is not None:state=f'GET READY: {max(1,int(countdown-now)+1)}'
            elif recording:state=f"RECORDING {now-clip_start:.1f}/{settings['clip_seconds']:.1f}s"
            else:state='READY'
            for line,text in enumerate([label,f'{state}  |  Hands detected: {int(masks.sum())}  |  Clips saved: {saved}',note,'SPACE: record after 2-second countdown | Q / Esc: stop']):
                cv2.putText(overlay,text,(22,32+line*37),cv2.FONT_HERSHEY_SIMPLEX,.58,(218,241,224),1,cv2.LINE_AA)
            cv2.imshow(title,np.vstack([image,overlay]))
            remaining=max(1,int((1/settings['target_fps']-(time.monotonic()-loop))*1000))
            key=cv2.waitKey(remaining)&0xff
            if key in (ord('q'),27):break
            if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
            if key==32 and not a.preview and not recording and countdown is None:countdown=time.monotonic()+2
    finally:
        if detector:detector.close()
        cap.release();cv2.destroyAllWindows()
        print('Camera stopped. Partial clip discarded. Complete saved clips:',saved)
if __name__=='__main__':main()
