import numpy as np
from scipy.io import loadmat
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import csv, time

EPS=np.float32(1e-4)

def relation_col(v, nominal, sigma):
    v=np.asarray(v,dtype=np.float32)
    if nominal:
        return (v[:,None]==v[None,:]).astype(np.float32)
    # torch.std default correction=1 (sample standard deviation)
    radius=np.std(v,ddof=1,dtype=np.float64)/float(sigma)
    R=1.0-np.abs(v[:,None]-v[None,:])
    R[R < radius]=0.0
    return R.astype(np.float32)

def importance(R):
    n=R.shape[0]
    # exact authors' code: fe and a log-complement quantity they call fce in code
    fe=-np.mean(np.log2(R.sum(axis=1,dtype=np.float64)/n + 1e-4))
    ce_code=-np.mean(np.log2((1-R).sum(axis=1,dtype=np.float64)/n + 1e-4))
    return fe+ce_code

def fce_from_card(s,n):
    total_pairs=n*(n-1)/2.0
    return np.mean((total_pairs - s*(s-1)/2.0)/total_pairs)

def scale_components(R0, n, *, lam=0.0, delta=0.0, smooth=None):
    # exact original adds 1e-4 to every fused/single relation BEFORE weights, FCE, RFC, LOO.
    R=R0.astype(np.float32,copy=False)+EPS
    s=R.sum(axis=1,dtype=np.float32)
    total_pairs=n*(n-1)/2.0

    if lam==0.0:
        pair=s*(s-1)/2.0
    else:
        # P_lambda = P_card + lambda/2 sum mu(1-mu), applied to the effective R used by code.
        fuzz=(R.astype(np.float64)*(1.0-R.astype(np.float64))).sum(axis=1)
        pair=s*(s-1)/2.0 + lam*0.5*fuzz
    F=np.mean((total_pairs-pair)/total_pairs)

    # RFC exactly as code (symmetric relation)
    sum_full=float(s.sum(dtype=np.float32))
    diag=np.diag(R)
    sum_after=sum_full - 2.0*s + diag
    rnc=s - sum_after/(n-1)

    # Exact leave-one-out FCE, vectorized exploiting quadratic form.
    # For each deleted i, cardinalities for remaining rows are c_j=s_j-R[j,i].
    denx=(n-1)*(n-2)/2.0
    Sbase=np.sum(s*s-s)
    # dot_i = sum_j s_j R_ji
    dot=R.T @ s
    colsum=R.sum(axis=0,dtype=np.float32)
    colsq=(R*R).sum(axis=0,dtype=np.float32)
    q_all=Sbase - 2.0*dot + colsq + colsum
    cdiag=s-diag
    q_excl=q_all - (cdiag*cdiag-cdiag)

    if lam==0.0:
        Fminus=1.0 - (q_excl/(2.0*denx))/(n-1)
    else:
        # extra fuzziness sum for rows j != i after deletion. Need mu-vector fuzziness for each remaining row,
        # with deleted membership removed. For row j: qf_j_full=sum_k R_jk(1-R_jk).
        RR=R
        fuzzrow=(RR*(1-RR)).sum(axis=1,dtype=np.float32)
        f_all=float(fuzzrow.sum()) - (RR*(1-RR)).sum(axis=0)  # column i removed from all rows
        # exclude deleted row i after also removing its diagonal membership contribution
        f_deleted_row=fuzzrow - diag*(1-diag)
        f_excl=f_all-f_deleted_row
        mean_pair=(q_excl/2.0 + lam*0.5*f_excl)/(n-1)
        Fminus=1.0-mean_pair/denx

    if delta>0:
        rf=np.maximum(0.0,(F-Fminus)/(F+delta))
    else:
        rf=np.clip(1.0-Fminus/F,0.0,1.0) if F!=0 else np.zeros(n)

    if smooth is None:
        g=np.where(rnc>0,(n-np.abs(rnc))/(2*n),np.sqrt((n+np.abs(rnc))/(2*n)))
    else:
        alpha,L,U,sn=smooth
        g=(U+L)/2.0 - (U-L)/2.0*np.tanh(alpha*rnc/sn)
    od=rf*g
    w=np.sqrt(s/n)
    return od,w,F,rnc,rf

def hfce_stream(X,sigma,fusion,lam=0.0,delta=0.0,smooth=None):
    X=np.asarray(X)
    n,m=X.shape
    ID=(X>=1).all(axis=0) & (X.max(axis=0)!=X.min(axis=0))
    imps=np.empty(m,float)
    sum_single=np.zeros(n,float)
    # first pass importance + singles
    for k in range(m):
        R=relation_col(X[:,k],bool(ID[k]),sigma)
        imps[k]=importance(R)
        od,w,*_=scale_components(R,n,lam=lam,delta=delta,smooth=smooth)
        sum_single+=(1-od)*w
    order=np.argsort(imps,kind='stable')
    # prefix fused relations in ascending-importance order. Each prefix length l corresponds to code k=m-l.
    pref=None
    subset_terms=[]
    for idx in order:
        R=relation_col(X[:,idx],bool(ID[idx]),sigma)
        if pref is None: pref=R.copy()
        elif fusion=='Min': np.minimum(pref,R,out=pref)
        elif fusion=='Max': np.maximum(pref,R,out=pref)
        elif fusion=='Multiply': pref*=R
        else: raise ValueError(fusion)
        od,w,*_=scale_components(pref,n,lam=lam,delta=delta,smooth=smooth)
        subset_terms.append((1-od)*w)
    # Code sums all m subsets; order doesn't matter for sum.
    sum_as=np.sum(np.vstack(subset_terms),axis=0)
    OS=1.0-(sum_single+sum_as)/(2*m)
    return OS,ID,imps,order

PARAMS={
'Aud':(0.4,'Max'),'Chess':(0.4,'Max'),'Lym':(0.4,'Min'),'Mush1':(0.4,'Min'),'Mush2':(0.4,'Min'),'Mush3':(0.4,'Min'),'Vote':(0.4,'Max'),
'Glass':(.4,'Multiply'),'Iono':(.5,'Min'),'Iris':(.4,'Min'),'Musk':(.2,'Min'),'Wave':(.2,'Min'),'Wbc':(1.4,'Multiply'),'Yeast':(.9,'Min'),
'Arr':(.3,'Multiply'),'Band1':(.3,'Multiply'),'Band2':(.3,'Min'),'German':(.3,'Max'),'Heart':(.3,'Max'),'Hepa':(.4,'Min')}
FILES={'Aud':'audiology_variant1','Chess':'chess_nowin_87_variant1','Lym':'lymphography','Mush1':'mushroom_p_221_variant1','Mush2':'mushroom_p_467_variant1','Mush3':'mushroom_p_85_variant1','Vote':'vote_republican_29_variant1','Glass':'glass','Iono':'ionosphere_b_24_variant1','Iris':'iris_Irisvirginica_11_variant1','Musk':'musk','Wave':'waveform_0_100_variant1','Wbc':'wbc_malignant_39_variant1','Yeast':'yeast_ERL_5_variant1','Arr':'arrhythmia_variant1','Band1':'bands_band_16_variant1','Band2':'bands_band_6_variant1','German':'german_1_14_variant1','Heart':'heart270_2_16_variant1','Hepa':'hepatitis_2_9_variant1'}
PUB={'Aud':.887,'Chess':.944,'Lym':1.0,'Mush1':.983,'Mush2':.928,'Mush3':.954,'Vote':.995,'Glass':.873,'Iono':1.0,'Iris':1.0,'Musk':1.0,'Wave':.763,'Wbc':.998,'Yeast':1.0,'Arr':.825,'Band1':.875,'Band2':.986,'German':.983,'Heart':.991,'Hepa':.999}

def load(name):
    A=loadmat(Path(__file__).parent/'Datasets'/(FILES[name]+'.mat'))['trandata']
    return A[:,:-1],A[:,-1].astype(int)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('names',nargs='*'); args=ap.parse_args()
    names=args.names or list(FILES)
    out=[]
    for name in names:
        X,y=load(name); sig,fu=PARAMS[name]; t=time.time()
        OS,ID,imps,order=hfce_stream(X,sig,fu)
        auc=roc_auc_score(y,OS); apv=average_precision_score(y,OS)
        row=[name,len(y),X.shape[1],int(y.sum()),int(ID.sum()),sig,fu,PUB[name],auc,auc-PUB[name],apv,time.time()-t]
        print(row,flush=True); out.append(row)
    with open('/mnt/data/hfce_exact_reproduction.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['dataset','n','m','outliers','ID_nominal_cols','sigma','fusion','published_auc','reproduced_auc','diff','AP','seconds']); w.writerows(out)
