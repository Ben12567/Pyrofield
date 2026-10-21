"""Independent Monte Carlo calibration of the residual autocorrelation screen.

Additional synthetic analyses; original manuscript figures and archived benchmark
records are not changed. The null design is U~Uniform(2,7), R0~Uniform(.06,.16),
four frames, 72 bearings, sigma/D=.03 and the stated fixed covariance. A
separate calibration ensemble sets a 1% upper-tail order statistic. Held-out
null and covariance-mismatch ensembles evaluate its error rates.
"""
import argparse,csv,json,math,time
from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar, least_squares
from scipy.stats import beta
import pyrofield_reduced as m

HERE=Path(__file__).resolve().parent
K=72
PSI=np.linspace(0,2*np.pi,K,endpoint=False)

def interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]

def statistic(r):
    r=r.reshape(4,K);r=r-r.mean(axis=1,keepdims=True)
    return float(np.mean(np.sum(r*np.roll(r,1,axis=1),axis=1)/np.sum(r*r,axis=1))*np.sqrt(4*K))

def one(rng,ell_true,ell_assumed,check_reference=False,check_legacy=False):
    R0,U=float(rng.uniform(.06,.16)),float(rng.uniform(2,7))
    Rh,L=m.head_rate(R0,U),float(m.L_anderson(U))
    times=1000/Rh*np.array([.25,.5,.75,1.])
    gen=m.BoundaryOperator(times,sigma=30,corr_length=1000*ell_true,Rh_ref=Rh,L_ref=L)
    ass=gen if ell_true==ell_assumed else m.BoundaryOperator(times,sigma=30,corr_length=1000*ell_assumed,Rh_ref=Rh,L_ref=L)
    inv=np.array([solve_triangular(c,np.eye(K),lower=True) for c in ass.chol])
    def whiten(x):return np.einsum('fij,fj->fi',inv,x.reshape(4,K),optimize=False).ravel()
    y=gen(Rh,L)+gen.sample_noise(rng);wy=whiten(y)
    def shape_fit(z):
        shape=1+np.exp(z)
        v=whiten(np.concatenate([m.front_radius(1.,shape,t,PSI) for t in times]))
        a=float(np.dot(v,wy)/np.dot(v,v))
        return a*v-wy,a
    def residual(z):return shape_fit(z)[0]
    def objective(z):
        r=residual(z);return float(r@r)
    fit=minimize_scalar(objective,bounds=(np.log(1e-5),np.log(7.)),method='bounded',options={'xatol':1e-10})
    if not fit.success:raise RuntimeError('Shape fit failed')
    r,Rh_hat=shape_fit(fit.x)
    L_hat=float(1+np.exp(fit.x));U_hat=m.anderson_inverse(L_hat)
    R0_hat=float(Rh_hat/(1+m.wind_factor(U_hat)))
    if not (.01<R0_hat<1 and .2<U_hat<m.U_C):
        raise RuntimeError('Fitted parameters outside the interior benchmark family')
    comparison={}
    if check_reference or check_legacy:
        def canonical(p):
            rh=m.head_rate(np.exp(p[0]),np.exp(p[1]));length=m.L_anderson(np.exp(p[1]))
            return whiten(ass(rh,length)-y)
    if check_reference:
        starts=[np.log([.1,2.]),np.log([.1,6.])]
        ref=min((least_squares(canonical,start,bounds=(np.log([.01,.2]),np.log([1.,20.])),
                              xtol=1e-12,ftol=1e-12,gtol=1e-12) for start in starts),key=lambda q:q.cost)
        cost_error=abs(float(r@r)-2*ref.cost)
        z_error=abs(statistic(r)-statistic(ref.fun))
        if cost_error>1e-6 or z_error>1e-4:
            raise RuntimeError(f'Reference fit disagreement: {cost_error}, {z_error}')
        comparison={'reference_cost_absolute_error':cost_error,'reference_z_absolute_error':z_error}
    if check_legacy:
        # Reproduce the original solve order, as LM can follow different local
        # paths after roundoff changes in a precomputed inverse multiplication.
        def legacy_residual(p):
            u=np.exp(p[1])
            model=np.concatenate([m.front_radius(m.head_rate(np.exp(p[0]),u),m.L_anderson(u),t,PSI) for t in times])
            return ass.whiten(model-y)
        legacy=least_squares(legacy_residual,np.log([.1,4.]),method='lm')
        comparison.update(legacy_z=statistic(legacy.fun),legacy_chi2=float(2*legacy.cost),
                          legacy_U_hat=float(np.exp(legacy.x[1])),legacy_success=bool(legacy.success))
    return {'R0_true':R0,'U_true':U,'ell_true':ell_true,'ell_assumed':ell_assumed,
            'z':statistic(r),'chi2':float(r@r),'L_hat':float(1+np.exp(fit.x)),
            'Rh_hat':Rh_hat,'U_hat':U_hat,'R0_hat':R0_hat,
            'shape_evaluations':int(fit.nfev),**comparison}

def archived_profile_comparison():
    with (HERE/'output/graded_benchmark_cases.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    out={}
    for name,q in [('all',rows),('correct_specification',[r for r in rows if r['cell'].startswith(('s0.05_','s0.1_','s0.2_','s0.3_'))])]:
        guard=[];inside=[];bounds=[]
        for r in q:
            u=float(r['U_hat']);g=float(r['DU_d30'])>=4 and r['inc_d30']=='False'
            ci=float(r['ci_lo'])>=u/1.3 and float(r['ci_hi'])<=u*1.3
            guard.append(g);inside.append(ci);bounds.append(r['inc_d30']=='True')
        out[name]={'n':len(q),'guard_c4_determined':sum(guard),'profile_c3_84_within_window':sum(inside),
                   'agreement':sum(a==b for a,b in zip(guard,inside)),
                   'guard_yes_interval_no':sum(a and not b for a,b in zip(guard,inside)),
                   'guard_no_interval_yes':sum(not a and b for a,b in zip(guard,inside)),
                   'note':'Thresholds differ: guard c=4, archived interval c=3.84. Bounds can downgrade guard decisions.'}
    return out

def capped_false_determinations():
    with (HERE/'output/guard_benchmark_cases.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    q=[r for r in rows if r['class']=='C3' and r['error_model']=='E2'
       and float(r['DU_GC_d30'])>=4 and r['GC_inconclusive_d30']=='False']
    return {'n':len(q),'cap_wind':m.U_C,
            'cases':[{'U_true':float(r['U_true']),'U_hat':float(r['U_hat']),
                      'upper_displaced_wind':float(r['U_hat'])*1.3,
                      'DU_c4':float(r['DU_GC_d30']),
                      'upper_displacement_on_plateau':float(r['U_hat'])*1.3>m.U_C} for r in q],
            'interpretation':'The sampled upper profile point is already on the capped plateau. A full profile displays this plateau but cannot ensure its inclusion in the nominal interval; noise can exclude the entire plateau.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--calibration',type=int,default=2000);p.add_argument('--validation',type=int,default=1000)
    p.add_argument('--out',type=Path,default=HERE/'output');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    rows=[];summary={};start=time.monotonic()
    check_rng=np.random.default_rng(2026102300)
    checks=[one(check_rng,true,assumed,check_reference=True)
            for assumed in (0.,.1) for true in (assumed,.3) for _ in range(12)]
    for j,assumed in enumerate((0.,.1)):
        cal_rng=np.random.default_rng(2026102100+j)
        cal=[]
        for i in range(args.calibration):
            r=one(cal_rng,assumed,assumed);r.update(split='calibration',case=i);cal.append(r['z']);rows.append(r)
        rank=int(math.ceil((args.calibration+1)*.99))
        if rank>args.calibration:raise ValueError('Calibration sample too small')
        threshold=float(np.sort(cal)[rank-1]);results=[]
        print(f'ell={assumed}: calibrated threshold {threshold:.6f}',flush=True)
        for h,(name,true) in enumerate((('matched',assumed),('mismatched',.3))):
            rng=np.random.default_rng(2026102200+j*10+h)
            stats=[]
            for i in range(args.validation):
                r=one(rng,true,assumed);r.update(split='validation_'+name,case=i);rows.append(r);stats.append(r['z'])
            count=int(np.sum(np.asarray(stats)>threshold));old=int(np.sum(np.asarray(stats)>2.33))
            results.append({'design':name,'ell_true':true,'n':args.validation,'calibrated_flagged':count,
                            'calibrated_rate':count/args.validation,'cp95':interval(count,args.validation),
                            'legacy_z2_33_flagged':old,'median_z':float(np.median(stats))})
            print(f'ell={assumed}, {name}: {count}/{args.validation} calibrated, {old} legacy',flush=True)
        summary[str(assumed)]={'calibration_n':args.calibration,'calibration_seed':2026102100+j,'rank_1based':rank,
                               'threshold':threshold,'validation':results}
    out={'design':'Additional parametric Monte Carlo calibration, independent validation; known covariance conditional on generating geometry.',
         'training_design':{'U_range':[2,7],'R0_range':[.06,.16],'frames':4,'bearings':72,'sigma_over_D':.03},
         'nominal_tail_probability':.01,'cases_total':len(rows),'settings':summary,
         'archived_profile_comparison':archived_profile_comparison(),
         'capped_false_determinations':capped_false_determinations(),
         'reference_fit_checks':{'n':len(checks),'seed':2026102300,
             'max_cost_absolute_error':max(r['reference_cost_absolute_error'] for r in checks),
             'max_z_absolute_error':max(r['reference_z_absolute_error'] for r in checks),
             'method':'Independent two-parameter bounded least squares in (log R0, log U), two starts; comparison cases excluded from all calibration and validation rates.'},
         'elapsed_seconds':time.monotonic()-start,
         'limitations':'Design-averaged calibration, not a per-fire fitted-model bootstrap or a validation on observed operational fires.'}
    with (args.out/'residual_calibration_cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (args.out/'residual_calibration.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
