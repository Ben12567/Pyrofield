"""Read-only checks of archived records and selected analytical identities.

This is a focused reproducibility check, not a rerun of the full benchmark.
Run from any directory: python code/validate_deposit.py
"""
from pathlib import Path
from collections import Counter
import ast
import csv
import hashlib
import json
import sys

import numpy as np
from scipy.stats import beta

import pyrofield_reduced as m
from identifiability_extensions import sd_logU
from rothermel import validate

HERE = Path(__file__).resolve().parent
OUT = HERE/'output'

def rows(name):
    with (OUT/name).open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def main():
    for p in HERE.rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    for p in HERE.rglob('*.json'):
        json.loads(p.read_text(encoding='utf-8'))
    labelled = rows('guard_benchmark_cases.csv')
    graded = rows('graded_benchmark_cases.csv')
    mismatch = rows('mismatch_cases.csv')
    assert [len(labelled),len(graded),len(mismatch)] == [1000,1140,780]
    correct = [r for r in graded if r['cell'].startswith(('s0.05_', 's0.1_', 's0.2_', 's0.3_'))]
    assert len(correct)==720
    covered=sum(r['covers']=='True' for r in correct)
    assert covered==684
    table=json.loads((OUT/'benchmark_tables.json').read_text())
    assert table['graded_correct_spec']['coverage']==[covered,len(correct)]
    paired = labelled+graded
    assert all(0<float(r['U_true'])<float('inf') and 0<float(r['U_hat'])<float('inf') for r in paired)
    # Apply the final bound-aware rule to the archived labelled records.
    structural={}
    for error_model in ('E1','E2'):
        q=[r for r in labelled if r['error_model']==error_model and r['class'] in ('C1','C2','C3')]
        n=sum(float(r['DU_GC_d30'])>=4 and r['GC_inconclusive_d30']=='False' for r in q)
        assert [len(q),n]==[300,0 if error_model=='E1' else 6]
        structural[error_model]={'n':len(q),'false_determinations':n}
    anna=json.loads((OUT/'annaburroo_demo.json').read_text())['summary']
    assert [anna[k] for k in ('n_fires','n_plots','n_isochrones')]==[7,5,13]
    assert anna['plot_level']['mf']['n_plots_log_s_positive']==5
    assert anna['coverage']['covers_fixed'][:2]==[1,7]
    # The compensation family must preserve the complete front, not just head rate.
    U0,R0,k=4.,.1,.366
    Rh,L=m.head_rate(R0,U0),m.L_linear(U0,k)
    psi=np.linspace(0,2*np.pi,73)
    ref=m.front_radius(Rh,L,500,psi)
    max_front_error=0.
    for u in (2.,3.,5.,7.):
        r0=Rh/(1+m.wind_factor(u)); ku=(L-1)/u
        front=m.front_radius(m.head_rate(r0,u),m.L_linear(u,ku),500,psi)
        max_front_error=max(max_front_error,float(np.max(np.abs(front-ref))))
    assert max_front_error<1e-9
    floor_errors=[]
    for u in (3.,5.):
        ex=u*m.dL_anderson_raw(u)/(m.L_anderson_raw(u)-1)
        pos=sd_logU(u)
        for sigma in (.1,.3):
            numerical=sd_logU(u,sigma_s=sigma)
            expected=np.sqrt(pos**2+(sigma/ex)**2)
            floor_errors.append(abs(numerical-expected)/expected)
    assert max(floor_errors)<1e-5
    validation=validate()
    max_ros_error=max(abs(ros/ref-1) for _,ros,ref,_,_ in validation)
    assert max_ros_error<.10
    manifest=HERE.parent/'MANIFEST.sha256'
    checked=0
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            digest,name=line.split('  ',1)
            assert hashlib.sha256((HERE.parent/name).read_bytes()).hexdigest()==digest,name
            checked+=1
    print(json.dumps({
        'status':'PASS','python':sys.version.split()[0],
        'archived_cases':{'labelled':1000,'graded':1140,'additional_mismatch':780},
        'figure3_paired_records':len(paired),'correct_specification_coverage':[covered,720],
        'coverage_cp95':[float(beta.ppf(.025,covered,720-covered+1)),float(beta.ppf(.975,covered+1,720-covered))],
        'structural_cases':structural,'experimental_counts':[7,5,13],
        'compensation_max_front_error_m':max_front_error,
        'precision_floor_max_relative_error':max(floor_errors),
        'rothermel_max_relative_ros_error':max_ros_error,
        'manifest_files_verified':checked,
        'scope':'Archived records plus focused analytical checks; no full simulation rerun.'
    },indent=2))

if __name__=='__main__': main()
