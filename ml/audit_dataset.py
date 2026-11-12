"""Inspect REAL clips; do not fabricate samples or promote frames to independent recordings."""
from pathlib import Path
import argparse,json,collections,hashlib
import numpy as np
from vision.landmarks import FEATURE_VERSION,FEATURE_COUNT
from vision.storage import clip_quality

def audit(root,mapping_path):
    mapping=json.loads(Path(mapping_path).read_text());counts=collections.Counter();groups={};errors=[];hashes={};clips=[]
    for path in sorted(Path(root).rglob('*.npz')):
        try:
            with np.load(path,allow_pickle=False) as data:
                meta=json.loads(str(data['metadata'].item()));points=data['points'];masks=data['masks'];features=data['features'];times=data['timestamps_ms']
                n=len(points);label=meta['label'];spec=mapping['labels'][label]
                assert points.shape==(n,2,21,3) and features.shape==(n,FEATURE_COUNT) and masks.shape==(n,2) and times.shape==(n,), 'Array shape mismatch'
                assert n>0 and np.isfinite(points).all() and np.isfinite(features).all(), 'Nonfinite or empty arrays'
                assert np.isin(masks,[0,1]).all() and (np.diff(times)>0).all(),'Invalid masks or time order'
                assert np.array_equal(features[:,-2:],masks),'Presence feature does not match masks'
                assert meta['feature_version']==FEATURE_VERSION,'Feature version mismatch'
                assert meta['mapping_version']==mapping['mapping_version'] and spec['reviewed'] and meta['mapping_reviewed'],'Unreviewed or incompatible mapping'
                assert meta['gesture_kind']==spec['kind'] and meta['expected_hands']==spec['expected_hands'],'Gesture definition changed'
                assert meta['consent_confirmed'] and not meta['raw_video_stored'],'Consent or privacy metadata invalid'
                assert path.parent.name==label and path.parent.parent.name==meta['session_id'] and path.parent.parent.parent.name==meta['signer_id'],'Path and metadata differ'
                quality=clip_quality(masks,spec['expected_hands'])
                assert quality['accepted'],'Insufficient hand visibility/frames'
                digest=hashlib.sha256(points.astype(np.float32).tobytes()+masks.astype(np.uint8).tobytes()).hexdigest()
                assert digest==meta['content_sha256'],'Content checksum mismatch'
                assert digest not in hashes,'Exact duplicate clip: '+hashes.get(digest,'')
                hashes[digest]=str(path)
                signer=meta['signer_id'];session=meta['session_id'];counts[label]+=1
                groups.setdefault(signer,{}).setdefault(session,collections.Counter())[label]+=1
                clips.append({'path':str(path),'label':label,'signer_id':signer,'session_id':session,'kind':spec['kind'],'frames':n,'sha256':digest,'lighting':meta['lighting'],'angle':meta['angle'],'distance':meta['distance']})
        except Exception as e:errors.append({'path':str(path),'error':str(e) or type(e).__name__})
    reviewed=[label for label,spec in mapping['labels'].items() if spec['reviewed']]
    static=[label for label in reviewed if mapping['labels'][label]['kind']=='static']
    # Conservative suggested pilot gate: each class in >=3 signers, >=2 sessions/signer, >=5 independent clips/session.
    coverage={}
    for label in reviewed:
        qualified=[signer for signer,sessions in groups.items() if sum(1 for values in sessions.values() if values[label]>=5)>=2]
        coverage[label]={'clips':counts[label],'qualified_signers':qualified,'pilot_coverage_ready':len(qualified)>=3}
    ready=bool(static) and len(static)>=3 and not errors and all(coverage[label]['pilot_coverage_ready'] for label in static)
    return {'schema_version':1,'valid_clips':len(clips),'invalid_clips':errors,'counts_by_label':dict(counts),'reviewed_labels':reviewed,'unreviewed_target_labels':[x for x in mapping['labels'] if x not in reviewed],'static_labels':static,'coverage':coverage,'groups':groups,'pilot_training_data_ready':ready,'all_26_reviewed':len(reviewed)==26,'all_26_coverage_ready':len(reviewed)==26 and all(c['pilot_coverage_ready'] for c in coverage.values()),'training_performed':False,'clips':clips,'limitations':'Pilot coverage thresholds are collection targets, not a guarantee of recognition quality. Split by signer when possible; otherwise report session-only evaluation explicitly.'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='data/raw');p.add_argument('--mapping',default='config/alphabet.json');p.add_argument('--output',default='reports/dataset_audit.json');a=p.parse_args()
    report=audit(a.root,a.mapping);target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2))
    print('Valid clips:',report['valid_clips']);print('Invalid clips:',len(report['invalid_clips']));print('Counts:',report['counts_by_label']);print('Pilot data ready:',report['pilot_training_data_ready']);print('Full report:',target)
    if report['invalid_clips']:raise SystemExit(1)
if __name__=='__main__':main()
