#!/usr/bin/env python3
"""Reproduce printed MSLS-val headline, not full image inference (no GPU)."""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'msls_crossed_and_gates_v6/kit'))
from euot_v6_analysis import HitStore,did,query_test

def close(x,y):
 if abs(float(x)-float(y))>0.011:raise AssertionError(f'manuscript contrast mismatch: {x} != {y}')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('outputs/headline_msls.json'));args=ap.parse_args()
 H=HitStore()
 n6=H.add_v6_dir(ROOT/'msls_crossed_and_gates_v6/hits_v6')
 nu=H.add_archived(ROOT/'msls_setting_U/hit_arrays_v4','U','msls_')
 ns=H.add_archived(ROOT/'msls_setting_S/hit_arrays_v5','S','msls_v5_')
 assert (n6,nu,ns)==(288,36,36)
 paper={'U':{'arms':[45.09,46.40,45.50,49.73],'relax':0.41,'did':2.93,'ci':[1.17,4.82]},
        'RE':{'arms':[44.77,46.44,45.54,49.55],'relax':0.77,'did':2.34,'ci':[0.54,4.28]},
        'UW':{'arms':[70.41,70.41,69.10,67.88],'relax':-1.31,'did':-1.22,'ci':[-3.74,1.35]},
        'S':{'arms':[70.14,71.44,69.86,67.75],'relax':-0.27,'did':-3.42,'ci':[-5.86,-0.99]}}
 out={'protocol':'740 paired MSLS-val queries; seed-averaged 42/108/2026; 2000 bootstrap draws, 10000 two-sided sign flips, numpy.default_rng(42)','settings':{}}
 for s,expected in paper.items():
  occ='occ50'
  arr={name:H.mean3(s,tr,gate,occ) for name,tr,gate in [('bal','bal','none'),('bal_g','bal','combined'),('uot','uot','none'),('uot_g','uot','combined')]}
  arms=[round(100*float(x.mean()),2) for x in arr.values()]
  for x,y in zip(arms,expected['arms']):close(x,y)
  relax=query_test((arr['uot']-arr['bal'])*100)
  interaction=query_test(did(H,s,occ))
  close(relax['diff_pp'],expected['relax']);close(interaction['diff_pp'],expected['did'])
  for x,y in zip(interaction['ci95'],expected['ci']):close(x,y)
  out['settings'][s]={'R1_pct':dict(zip(['Bal','Bal+G','UOT','UOT+G'],arms)),'relaxation_alone':relax,'combined_gate_DiD':interaction}
 # mass-only gate, share effect, SALAD released reference
 mass=query_test(did(H,'U','occ50','mass'))
 close(mass['diff_pp'],2.79)
 close(mass['ci95'][0],1.13);close(mass['ci95'][1],4.59)
 out['mass_only_unit_share_half_DiD']=mass
 dU,dS,dUW,dRE=(did(H,s,'occ50') for s in ('U','S','UW','RE'))
 main=query_test(0.5*((dUW-dU)+(dS-dRE)))
 close(main['diff_pp'],-4.95)
 # Source manuscript prints reduction as +4.95 pp; our convention is share 1/65 minus share 1/2.
 close(main['ci95'][0],-7.68);close(main['ci95'][1],-2.32)
 out['global_share_main_effect_1_65_minus_half']=main
 salad=np.load(ROOT/'msls_setting_U/hit_arrays_v4/msls_official_pretrained_salad_occ50_hits.npy',allow_pickle=False)[:,0].mean()*100
 close(round(float(salad),2),71.76)
 out['released_SALAD_R1_pct_at_occ50']=round(float(salad),2)
 target=args.output if args.output.is_absolute() else ROOT/args.output
 target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(out,indent=2)+'\n')
 print('PASS: 4 MSLS severe-occlusion factorials, 4 ungated relaxations, mass-only gate, crossed global-share contrast and SALAD reference; output',target)
 for s in ('U','RE','UW','S'):print(f"{s}: R@1 {out['settings'][s]['R1_pct']} DiD {out['settings'][s]['combined_gate_DiD']}")
 print('global-share main effect (1/65 minus 1/2):',main)
if __name__=='__main__':main()
