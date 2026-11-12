"""Download Google's hand detector; this is NOT a trained ISL classifier."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,zipfile
URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='models/hand_landmarker.task');args=parser.parse_args()
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if not zipfile.is_zipfile(target):raise SystemExit('Existing asset is invalid; rename it before retrying.')
        print('Detector already present:',target);return
    temporary=target.with_suffix('.download')
    try:
        with urllib.request.urlopen(URL,timeout=90) as response,temporary.open('wb') as out:
            while chunk:=response.read(1024*1024):out.write(chunk)
        if temporary.stat().st_size<100000 or not zipfile.is_zipfile(temporary):raise ValueError('Invalid detector download')
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:raise ValueError('Detector ZIP integrity check failed')
        temporary.replace(target)
    finally:
        if temporary.exists():temporary.unlink()
    receipt={'source':URL,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'purpose':'Hand landmark detection only; not ISL alphabet recognition'}
    target.with_suffix('.source.json').write_text(json.dumps(receipt,indent=2))
    print('Downloaded hand detector:',target)
    print('SHA256:',receipt['sha256'])
    print('No ISL alphabet classifier has been trained.')
if __name__=='__main__':main()
