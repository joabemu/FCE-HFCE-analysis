import numpy as np, csv, time
from pathlib import Path
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, average_precision_score
from reproduce_hfce import relation_col, importance, FILES, PARAMS, PUB

EPS=np.float32(1e-4)
VARIANTS=['HFCE','PAIR1','REG1e-3','REG1e-2','SMOOTH4','PAIR1_SMOOTH4']

def core_quantities(R0,n):
    R=(R0.astype(np.float32,copy=False)+EPS)
    s=R.sum(axis=1,dtype=np.float32)
    diag=np.diag(R)
    total_pairs=n*(n-1)/2.0
    basepair=s*(s-1)/2.0
    F0=float(np.mean((total_pairs-basepair)/total_pairs))
    # fuzziness on underlying proper memberships, before eps
    mu=R0.astype(np.float32,copy=False)
    fuzzrow=(mu*(1-mu)).sum(axis=1,dtype=np.float32)
    F1=float(np.mean((total_pairs-(basepair+0.5*fuzzrow))/total_pairs))
    sum_full=float(s.sum(dtype=np.float32))
    sum_after=sum_full-2*s+diag
    rnc=s-sum_after/(n-1)
    denx=(n-1)*(n-2)/2.0
    Sbase=np.sum(s*s-s,dtype=np.float64)
    dot=R.T @ s
    colsum=R.sum(axis=0,dtype=np.float32)
    colsq=(R*R).sum(axis=0,dtype=np.float32)
    q_all=Sbase-2*dot+colsq+colsum
    cdiag=s-diag
    q_excl=q_all-(cdiag*cdiag-cdiag)
    base_mean_pair=q_excl/(2*(n-1))
    Fm0=1.0-base_mean_pair/denx
    # leave-one-out fuzziness correction from R0
    fcol=(mu*(1-mu)).sum(axis=0,dtype=np.float32)
    fall=float(fuzzrow.sum(dtype=np.float32))-fcol
    f_deleted_row=fuzzrow-np.diag(mu)*(1-np.diag(mu))
    fexcl=fall-f_deleted_row
    Fm1=1.0-(base_mean_pair+0.5*fexcl/(n-1))/denx
    w=np.sqrt(s/n)
    return F0,Fm0,F1,Fm1,rnc,w

def rf(F,Fm,delta=0):
    if delta>0: return np.maximum(0,(F-Fm)/(F+delta))
    if F==0: return np.zeros_like(Fm)
    return np.clip(1-Fm/F,0,1)

def g_orig(r,n):
    return np.where(r>0,(n-np.abs(r))/(2*n),np.sqrt((n+np.abs(r))/(2*n)))

def g_smooth(r,n,alpha=4):
    L=1/(2*n); U=np.sqrt(1-1/n); sn=n
    return (U+L)/2-(U-L)/2*np.tanh(alpha*r/sn)

def terms(R0,n):
    F0,Fm0,F1,Fm1,r,w=core_quantities(R0,n)
    g0=g_orig(r,n); gs=g_smooth(r,n)
    rf0=rf(F0,Fm0); rf1=rf(F1,Fm1)
    ods={
      'HFCE':rf0*g0,
      'PAIR1':rf1*g0,
      'REG1e-3':rf(F0,Fm0,1e-3)*g0,
      'REG1e-2':rf(F0,Fm0,1e-2)*g0,
      'SMOOTH4':rf0*gs,
      'PAIR1_SMOOTH4':rf1*gs,
    }
    return {k:(1-v)*w for k,v in ods.items()}

def run(X,sigma,fusion):
    X=np.asarray(X); n,m=X.shape
    ID=(X>=1).all(axis=0)&(X.max(axis=0)!=X.min(axis=0))
    imps=np.empty(m)
    sums={v:np.zeros(n,float) for v in VARIANTS}
    # single scales + importance
    for k in range(m):
        R=relation_col(X[:,k],bool(ID[k]),sigma)
        imps[k]=importance(R)
        tt=terms(R,n)
        for v in VARIANTS: sums[v]+=tt[v]
    order=np.argsort(imps,kind='stable')
    pref=None
    for idx in order:
        R=relation_col(X[:,idx],bool(ID[idx]),sigma)
        if pref is None: pref=R.copy()
        elif fusion=='Min': np.minimum(pref,R,out=pref)
        elif fusion=='Max': np.maximum(pref,R,out=pref)
        else: pref*=R
        tt=terms(pref,n)
        for v in VARIANTS: sums[v]+=tt[v]
    return {v:1-sums[v]/(2*m) for v in VARIANTS}

def load(name):
    A=loadmat(Path(__file__).parent/'Datasets'/(FILES[name]+'.mat'))['trandata']
    return A[:,:-1],A[:,-1].astype(int)

if __name__=='__main__':
 import sys
 names=sys.argv[1:] or list(FILES)
 rows=[]
 for name in names:
    X,y=load(name); sig,fu=PARAMS[name]; t=time.time(); scores=run(X,sig,fu)
    row={'dataset':name,'published_auc':PUB[name],'n':len(y),'m':X.shape[1]}
    for v in VARIANTS:
       row[v+'_AUC']=roc_auc_score(y,scores[v]); row[v+'_AP']=average_precision_score(y,scores[v])
    row['seconds']=time.time()-t; rows.append(row); print(row,flush=True)
 with open('/mnt/data/hfce_variant_results.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
