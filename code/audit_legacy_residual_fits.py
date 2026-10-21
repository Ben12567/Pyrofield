"""Refit the 240 original residual-check observations without changing their archive.

The original seeds and noise draws are reused exactly. The legacy one-start LM
solution is compared with amplitude-profiled bounded shape optimisation. These
are paired numerical checks of old cases, not additional independent trials.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from residual_calibration import one

HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=HERE/'output');args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    original=json.loads((HERE/'output/misspecification_checks.json').read_text())['residual_diagnostic']
    settings={'true_ell0.3_assumed_iid':(.3,0.),'true_ell0.1_assumed_0.1':(.1,.1),
              'true_ell0.3_assumed_0.1':(.3,.1),'true_iid_assumed_iid':(0.,0.)}
    rows=[];summaries={}
    for name,(true,assumed) in settings.items():
        seed=31337+int(100*true)+int(1000*assumed)
        rng=np.random.default_rng(seed);group=[]
        for i in range(60):
            r=one(rng,true,assumed,check_legacy=True)
            r.update(cell=name,case=i,legacy_excess_chi2=r['legacy_chi2']-r['chi2'])
            rows.append(r);group.append(r)
        legacy=sum(r['legacy_z']>2.33 for r in group);profiled=sum(r['z']>2.33 for r in group)
        assert legacy==original[name]['flagged_z>2.33'],(name,legacy,original[name])
        summaries[name]={'n':60,'seed':seed,'legacy_flagged':legacy,'profiled_flagged':profiled,
            'legacy_cost_exceeds_profiled_by_gt1e_5':sum(r['legacy_excess_chi2']>1e-5 for r in group),
            'legacy_flagged_but_profiled_unflagged':sum(r['legacy_z']>2.33 and r['z']<=2.33 for r in group),
            'max_legacy_excess_chi2':max(r['legacy_excess_chi2'] for r in group),
            'max_profiled_z':max(r['z'] for r in group)}
    out={'scope':'Paired refit audit of the same 240 original observations; not independent new validation cases.',
         'legacy_flag_counts_match_archive':True,'settings':summaries,
         'interpretation':'A failed search for the minimum can produce residual correlation. Separate optimiser checks from threshold calibration; neither diagnoses shape-law error that is exactly compensated.'}
    with (args.out/'residual_legacy_fit_audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.out/'residual_legacy_fit_audit.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
