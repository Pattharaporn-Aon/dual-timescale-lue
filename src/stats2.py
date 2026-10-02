import numpy as np, json
from scipy.stats import chi2
from stats import *
S=json.load(open("results/stats_results.json")); ell=S["ell_gls"]; xg=np.array([S["theta_gls"][k] for k in NAMES])
Lg,ldg=whitener(ell); I=np.eye(n)
def profile(Lc,logdet,xopt,grid,ki):
    free=np.array([i for i in range(7) if i!=ki]); _,rs0=gls_fit(xopt.copy(),Lc); base=n*np.log(rs0/n)
    dev=[];xw=xopt.copy()
    for gv in grid:
        best=None
        for st in [np.r_[xopt[:ki],gv,xopt[ki+1:]], np.r_[xw[:ki],gv,xw[ki+1:]]]:
            xf,rs=gls_fit(st.copy(),Lc,free)
            if best is None or rs<best[1]: best=(xf,rs)
        xw=best[0]; dev.append(n*np.log(best[1]/n)-base)
    dev=np.array(dev); return dev-min(0,dev.min())
if __name__=="__main__":
    grid=np.geomspace(0.006,0.1,40); ki=1
    d_ou=profile(Lg,ldg,xg,grid,ki)
    x_ols,_=gls_fit(xg.copy(),I); d_iid=profile(I,0,x_ols,grid,ki)
    c=chi2.ppf(0.95,1)
    for name,dv in [("OU",d_ou),("iid",d_iid)]:
        ins=grid[dv<=c]; print(name,"tau CI",round(1/ins.max(),1),round(1/ins.min(),1),"open_low_tau",dv[-1]<=c, "min dev at tau",round(1/grid[dv.argmin()],1), "dev at b0=0.1",round(dv[-1],2))
    json.dump(dict(grid=grid.tolist(),dev_ou=d_ou.tolist(),dev_iid=d_iid.tolist(),x_ols=x_ols.tolist()),open("results/stats_b0profile.json","w"))
