"""Export the exact archived coordinates and bin counts used in Figure 3b.

No interpolation, jitter, synthetic points or generative image model is used.
Run: python code/prepare_figure3_records.py --out figure3_source
This exports source tables, not the editable PowerPoint layout.
"""
from pathlib import Path
import argparse, csv, json
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('figure3_source'))
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    base=Path(__file__).resolve().parent/'output'
    records=[]
    for source,name in [('labelled','guard_benchmark_cases.csv'),('graded','graded_benchmark_cases.csv')]:
        with (base/name).open(newline='',encoding='utf-8') as f:
            for i,r in enumerate(csv.DictReader(f),1):
                records.append({'source':source,'source_row':i,'U_true':float(r['U_true']),'U_hat':float(r['U_hat'])})
    assert len(records)==2140
    lo,hi=np.log10(.3),np.log10(20.)
    xy=np.log10([[r['U_true'],r['U_hat']] for r in records])
    h,xe,ye=np.histogram2d(xy[:,0],xy[:,1],bins=64,range=((lo,hi),(lo,hi)))
    assert h.sum()==2140
    ix=np.clip(np.floor((xy[:,0]-lo)/(hi-lo)*64).astype(int),0,63)
    iy=np.clip(np.floor((xy[:,1]-lo)/(hi-lo)*64).astype(int),0,63)
    for r,i,j in zip(records,ix,iy): r['cell_count']=int(h[i,j])
    with (args.out/'Figure3b_coordinates.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    meta={'n':2140,'units':'m s^-1','bins':64,'log10_range':[float(lo),float(hi)],'max_cell_count':int(h.max()),
          'scope':'Pooled descriptive display across assumptions; not a standalone accuracy or coverage estimate.'}
    (args.out/'Figure3b_metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta))

if __name__=='__main__': main()
