#!/usr/bin/env python3
"""Validate release SHA-256, original setting manifests, hit-matrix shape, and disallowed files."""
import hashlib, json, re, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fail(x):raise SystemExit('FAIL: '+x)
listed={}
for line in (ROOT/'SHA256SUMS.txt').read_text().splitlines():
 if not line.strip():continue
 digest,path=line.split('  ',1)
 if path in listed:fail('duplicate '+path)
 listed[path]=digest
IGNORED_DIRS={'.git','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','outputs','.venv'}
files={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt' and not IGNORED_DIRS.intersection(p.relative_to(ROOT).parts)}
if files!=set(listed):fail(f'inventory mismatch: missing={sorted(files-set(listed))[:5]} extra={sorted(set(listed)-files)[:5]}')
for name,digest in listed.items():
 if sha(ROOT/name)!=digest:fail('hash mismatch '+name)
n_man=0
for folder in ('msls_setting_U','msls_setting_S','msls_crossed_and_gates_v6'):
 base=ROOT/folder;manifest=json.loads((base/'SHA256_MANIFEST.json').read_text())
 if folder!='msls_crossed_and_gates_v6':
  iterator=((f'{d}/{name}',rec['sha256']) for d,entries in manifest.items() for name,rec in entries.items())
 else:iterator=manifest.items()
 for relative,digest in iterator:
  p=base/relative
  if not p.is_file() or sha(p)!=digest:fail('source manifest mismatch '+str(p))
  n_man+=1
nums={'v4':0,'v5':0,'v6':0}
for folder,sub,kind in [('msls_setting_U','hit_arrays_v4','v4'),('msls_setting_S','hit_arrays_v5','v5'),('msls_crossed_and_gates_v6','hits_v6','v6')]:
 for p in (ROOT/folder/sub).glob('*.npy'):
  a=np.load(p,allow_pickle=False)
  if a.shape!=(740,3) or a.dtype!=np.uint8 or not np.isin(a,[0,1]).all():fail('invalid hit matrix '+str(p))
  nums[kind]+=1
if nums!={'v4':42,'v5':36,'v6':288}:fail(f'unexpected hit counts {nums}')
for p in (ROOT/'msls_crossed_and_gates_v6/results').glob('*.npz'):
 with np.load(p,allow_pickle=False) as a:
  if not a.files:fail('empty npz '+str(p))
for p in ROOT.rglob('*'):
 if not p.is_file() or IGNORED_DIRS.intersection(p.relative_to(ROOT).parts):continue
 relative=p.relative_to(ROOT).as_posix()
 if p.suffix in ('.ipynb','.log','.tex','.pem','.key') or 'marker' in p.parts or 'notebook' in relative.lower() or p.name.endswith('.zip'):
  fail('disallowed filename '+relative)
 if p.suffix in ('.py','.json','.md','.txt','.cff') and p.name!='verify_release.py':
  s=p.read_text(errors='replace')
  for pat in (r'/content/drive/',r'/(?:home|Users)/[^/\s]+',r'/tmp/[A-Za-z_]*storage',r'/mnt/data/',r'!{4,}[A-Za-z ]+',r'ghp_[A-Za-z0-9]{20,}',r'sk-[A-Za-z0-9]{20,}'):
   if re.search(pat,s):fail('private pattern '+relative+' '+pat)
print(f'PASS: SHA-256 inventory {len(listed)} files; original settings {n_man} digest entries; hit matrices '+str(nums)+'; no prohibited filenames/locations')
