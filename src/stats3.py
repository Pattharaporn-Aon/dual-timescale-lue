import numpy as np, json
from scipy.stats import chi2
from stats import *
from stats2 import profile, Lg, ldg, xg
c=chi2.ppf(0.95,1); out={}
G={"alpha":np.geomspace(0.002,0.06,30),"phi":np.linspace(0,1,26),"r":np.linspace(0,0.005,26),"c0":np.linspace(0.2,1.0,26),"bg":np.linspace(0,0.6,25),"L0":np.geomspace(0.1,8,25)}
for k,g in G.items():
    ki=NAMES.index(k); dv=profile(Lg,ldg,xg,g,ki); ins=g[dv<=c]
    out[k]=dict(grid=g.tolist(),dev=dv.tolist(),mle=float(xg[ki]),ci=[float(ins.min()),float(ins.max())],hits_lower=bool(dv[0]<=c),hits_upper=bool(dv[-1]<=c))
    print(k,round(xg[ki],5),np.round(out[k]["ci"],5),out[k]["hits_lower"],out[k]["hits_upper"],flush=True)
json.dump(out,open("results/stats_profiles_full.json","w"),indent=1)
