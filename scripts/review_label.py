"""Record a HUMAN review; this script cannot verify sign-language correctness."""
import argparse,json
from pathlib import Path
from vision.storage import safe_id

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label',required=True,choices=list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    p.add_argument('--kind',required=True,choices=['static','dynamic'])
    p.add_argument('--hands',required=True,type=int,choices=[1,2])
    p.add_argument('--reference',required=True,help='Reference URL or local expert review note')
    p.add_argument('--reviewer',required=True,help='Pseudonymous reviewer ID')
    p.add_argument('--notes',default='')
    p.add_argument('--confirm-reviewed',action='store_true')
    a=p.parse_args()
    if not a.confirm_reviewed:raise SystemExit('Review this exact gesture with an ISL-competent person, then supply --confirm-reviewed.')
    safe_id(a.reviewer)
    if not a.reference.strip():raise SystemExit('A reference or review note is required.')
    path=Path('config/alphabet.json');data=json.loads(path.read_text())
    row=data['labels'][a.label]
    if row['reviewed']:raise SystemExit('Already reviewed. Do not change an existing mapping while clips use it. Create a new mapping version deliberately.')
    row.update(kind=a.kind,expected_hands=a.hands,reference_url=a.reference.strip(),reviewer_id=a.reviewer,reviewed=True,notes=a.notes)
    path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    print('Recorded human review for',a.label,'as',a.kind,'using',a.hands,'hand(s). No automatic linguistic verification is claimed.')
if __name__=='__main__':main()
