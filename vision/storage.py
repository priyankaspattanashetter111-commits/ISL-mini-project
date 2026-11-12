"""Versioned clips contain landmarks and quality metadata, never raw images."""
from pathlib import Path
import json,re,uuid,hashlib
from datetime import datetime,timezone
import numpy as np
from vision.landmarks import FEATURE_VERSION,FEATURE_COUNT

def safe_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}",value):
        raise ValueError("IDs must have 1..40 letters/digits/underscore/hyphen; no names or paths")
    return value

def clip_quality(masks,expected_hands,min_frames=20,min_valid_ratio=.85):
    masks=np.asarray(masks)
    if masks.ndim!=2 or masks.shape[1]!=2:raise ValueError("Invalid hand mask sequence")
    valid=(masks.sum(axis=1)>=expected_hands) if len(masks) else np.zeros(0,dtype=bool)
    ratio=float(valid.mean()) if len(valid) else 0.0
    accepted=len(masks)>=min_frames and ratio>=min_valid_ratio
    return {'frames':len(masks),'valid_ratio':round(ratio,4),'accepted':accepted}

def save_clip(root,signer,session,label,points,masks,features,timestamps,metadata):
    safe_id(signer);safe_id(session)
    if len(label)!=1 or label not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':raise ValueError("Label must be A-Z")
    points=np.asarray(points,dtype=np.float32);masks=np.asarray(masks,dtype=np.uint8)
    features=np.asarray(features,dtype=np.float32);timestamps=np.asarray(timestamps,dtype=np.int64)
    n=len(points)
    if n<1 or points.shape!=(n,2,21,3) or masks.shape!=(n,2) or features.shape!=(n,FEATURE_COUNT) or timestamps.shape!=(n,):raise ValueError("Inconsistent clip arrays")
    if not np.isfinite(points).all() or not np.isfinite(features).all() or not np.isin(masks,[0,1]).all():raise ValueError("Invalid clip values")
    if n>1 and not (np.diff(timestamps)>0).all():raise ValueError("Timestamps must increase")
    if not metadata.get('consent_confirmed'):raise ValueError("Consent must be confirmed")
    if not metadata.get('mapping_reviewed'):raise ValueError("Review the gesture mapping before saving approved clips")
    clip_id=uuid.uuid4().hex
    folder=Path(root)/safe_id(signer)/safe_id(session)/label;folder.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(points.tobytes()+masks.tobytes()).hexdigest()
    meta={**metadata,'schema_version':1,'feature_version':FEATURE_VERSION,'clip_id':clip_id,'label':label,'signer_id':signer,'session_id':session,'created_at':datetime.now(timezone.utc).isoformat(),'content_sha256':digest,'raw_video_stored':False}
    path=folder/(clip_id+'.npz')
    np.savez_compressed(path,points=points,masks=masks,features=features,timestamps_ms=timestamps,metadata=np.array(json.dumps(meta)))
    return path
