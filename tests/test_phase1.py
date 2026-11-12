import copy,json
import numpy as np
import pytest
from vision.landmarks import normalize,FEATURE_COUNT,unpack_result
from vision.storage import safe_id,clip_quality,save_clip
from ml.audit_dataset import audit

def hands():
    p=np.zeros((2,21,3),dtype=np.float32)
    for hand in range(2):
        p[hand,:,0]=np.linspace(.2,.4,21)+hand*.3
        p[hand,:,1]=np.linspace(.2,.6,21)
        p[hand,:,2]=np.linspace(0,.02,21)
    return p

def test_no_hands_are_explicit():
    assert np.array_equal(normalize(np.zeros((2,21,3)),[0,0],640,480),np.zeros(FEATURE_COUNT))
def test_translation_invariance():
    p=hands();q=p.copy();q[:,:,:2]+=.1
    np.testing.assert_allclose(normalize(p,[1,1],640,480),normalize(q,[1,1],640,480),atol=1e-5)
def test_scale_invariance():
    p=hands();np.testing.assert_allclose(normalize(p,[1,1],640,480),normalize(p*1.5,[1,1],640,480),atol=1e-5)
def test_relative_hand_position_is_preserved():
    p=hands();q=p.copy();q[1,:,0]+=.2
    assert not np.allclose(normalize(p,[1,1],640,480),normalize(q,[1,1],640,480))
def test_missing_hand_has_mask_and_zero_slot():
    feature=normalize(hands(),[1,0],640,480)
    assert np.array_equal(feature[63:126],np.zeros(63)) and feature[-2:].tolist()==[1,0]
def test_bad_coordinates_rejected():
    p=hands();p[0,0,0]=np.nan
    with pytest.raises(ValueError):normalize(p,[1,1],640,480)
def test_degenerate_hand_rejected():
    with pytest.raises(ValueError):normalize(np.zeros((2,21,3)),[1,0],640,480)
@pytest.mark.parametrize('value',['../p1','P 01','C:/tmp','', 'x'*41])
def test_invalid_ids_rejected(value):
    with pytest.raises(ValueError):safe_id(value)
def test_quality_requires_both_expected_hands():
    assert not clip_quality([[1,0]]*30,2)['accepted']
    assert clip_quality([[1,1]]*30,2)['accepted']
def test_quality_rejects_short_clip():assert not clip_quality([[1,1]]*5,2)['accepted']

def sample(tmp_path):
    p=np.stack([hands()+i*.0001 for i in range(30)]);mask=np.ones((30,2),dtype=np.uint8)
    f=np.stack([normalize(x,[1,1],640,480) for x in p]);times=np.arange(30)*67
    meta={'consent_confirmed':True,'mapping_reviewed':True,'mapping_version':'test-v1','gesture_kind':'static','expected_hands':2,'lighting':'normal','angle':'front','distance':'medium'}
    return p,mask,f,times,meta
def mapping(tmp_path):
    path=tmp_path/'map.json';path.write_text(json.dumps({'mapping_version':'test-v1','labels':{'A':{'reviewed':True,'kind':'static','expected_hands':2}}}));return path
def test_clip_round_trip_and_real_audit(tmp_path):
    args=sample(tmp_path);path=save_clip(tmp_path/'raw','P01','S01','A',*args)
    with np.load(path,allow_pickle=False) as data:assert data['points'].shape==(30,2,21,3)
    result=audit(tmp_path/'raw',mapping(tmp_path));assert result['valid_clips']==1 and not result['pilot_training_data_ready']
def test_duplicate_recordings_flagged(tmp_path):
    args=sample(tmp_path);save_clip(tmp_path/'raw','P01','S01','A',*args);save_clip(tmp_path/'raw','P01','S01','A',*args)
    result=audit(tmp_path/'raw',mapping(tmp_path));assert len(result['invalid_clips'])==1
def test_unconsented_sample_is_not_saved(tmp_path):
    args=list(sample(tmp_path));args[-1]['consent_confirmed']=False
    with pytest.raises(ValueError):save_clip(tmp_path/'raw','P01','S01','A',*args)
def test_out_of_order_timestamps_rejected(tmp_path):
    args=list(sample(tmp_path));args[3][2]=0
    with pytest.raises(ValueError):save_clip(tmp_path/'raw','P01','S01','A',*args)
def test_empty_dataset_is_not_ready(tmp_path):assert not audit(tmp_path/'raw',mapping(tmp_path))['pilot_training_data_ready']
