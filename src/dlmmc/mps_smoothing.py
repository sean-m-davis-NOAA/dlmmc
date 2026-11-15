"""Native float32 Metal Gaussian trend smoother with bounded draw workspace.

All panels remain together; GPU histories exist only for the active draw tile.
No sampled states, covariance floors, jitter or precision fallback are used.
"""
from functools import lru_cache
import os,time
import numpy as np
import torch
from .structured_mps import StructuredMPSTarget, state_layout, transition_code
from .tensor_target import model_components

SOURCE=r'''
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
constant uint D=12;
inline void mul(thread const float* a,thread const float* b,thread float* out){
 for(uint i=0;i<D;i++)for(uint j=0;j<D;j++){float v=0;for(uint k=0;k<D;k++)v+=a[i*D+k]*b[k*D+j];out[i*D+j]=v;}}
inline void mult(thread const float* a,thread const float* b,thread float* out){
 for(uint i=0;i<D;i++)for(uint j=0;j<D;j++){float v=0;for(uint k=0;k<D;k++)v+=a[i*D+k]*b[j*D+k];out[i*D+j]=v;}}
kernel void smooth(device const float* y,device const float* variance,device const bool* mask,
 device const float* F,device const float* gs,device const float* qs,device const float* cs,
 device float* hist,device float* out,device int* failure,
 constant uint& cells,constant uint& times,constant uint& count,constant uint& anchor,constant uint& varying_F,
 uint index [[thread_position_in_grid]]){
 if(index>=count)return;
 uint cell=index%cells,mat=index*144;
 // Filtered covariance later overwritten by smoothed covariance; full gain history.
 uint stride=times*(144+144+12),base=index*stride;
 device float* cov=hist+base;device float* gainhist=cov+times*144;device float* means=gainhist+times*144;
 device float* result=out+index*times*4;
 float P[144],G[144],Q[144],tmp[144],R[144],J[144],A[144],sc[144],next[144];
 float m[12],sm[12],mp[12],f[12],pf[12],k[12],v[12],scale[12];
 float B[144],bf[12];
 uint c0[12],c1[12];float g0[12],g1[12];
 failure[index]=-1;
 for(uint i=0;i<144;i++){P[i]=cs[mat+i];G[i]=gs[mat+i];Q[i]=qs[mat+i];}
 for(uint i=0;i<D;i++){
  m[i]=0;c0[i]=i;c1[i]=i;if(i==0)c1[i]=1;
  if(i==2||i==3){c0[i]=2;c1[i]=3;}if(i==4||i==5){c0[i]=4;c1[i]=5;}
  g0[i]=G[i*D+c0[i]];g1[i]=c0[i]==c1[i]?0:G[i*D+c1[i]];
 }
 for(uint t=0;t<times;t++){
  for(uint i=0;i<D;i++){
   mp[i]=g0[i]*m[c0[i]]+g1[i]*m[c1[i]];
   for(uint j=0;j<D;j++)tmp[i*D+j]=g0[i]*P[c0[i]*D+j]+g1[i]*P[c1[i]*D+j];
  }
  for(uint i=0;i<D;i++){
   m[i]=mp[i];for(uint j=0;j<D;j++)P[i*D+j]=Q[i*D+j]+tmp[i*D+c0[j]]*g0[j]+tmp[i*D+c1[j]]*g1[j];
  }
  uint at=cell*times+t;
  if(mask[at]){
   float s=variance[at],res=y[at];
   for(uint i=0;i<D;i++)f[i]=F[((varying_F ? index : cell)*times+t)*D+i];
   for(uint i=0;i<D;i++){
    float z=0;for(uint j=0;j<D;j++)z+=P[i*D+j]*f[j];pf[i]=z;s+=f[i]*z;res-=f[i]*m[i];
   }
   if(!isfinite(s)||s<=0){failure[index]=int(t);return;}
   for(uint i=0;i<D;i++){k[i]=pf[i]/s;m[i]+=k[i]*res;}
   for(uint j=0;j<D;j++){
    float z=0;for(uint h=0;h<D;h++)z+=f[h]*P[h*D+j];
    for(uint i=0;i<D;i++)B[i*D+j]=P[i*D+j]-k[i]*z;
   }
   for(uint i=0;i<D;i++){
    float z=0;for(uint j=0;j<D;j++)z+=B[i*D+j]*f[j];bf[i]=z;
   }
   for(uint i=0;i<D;i++)for(uint j=0;j<D;j++)P[i*D+j]=B[i*D+j]-bf[i]*k[j]+k[i]*k[j]*variance[at];
  }
  for(uint i=0;i<D;i++)means[t*D+i]=m[i];
  for(uint i=0;i<144;i++)cov[t*144+i]=P[i];
 }
 for(uint i=0;i<D;i++)sm[i]=m[i];for(uint i=0;i<144;i++)sc[i]=P[i];
 result[(times-1)*4]=sm[0];result[(times-1)*4+1]=sc[0];
 for(int t=int(times)-2;t>=0;t--){
  for(uint i=0;i<144;i++){P[i]=cov[t*144+i];next[i]=sc[i];}
  for(uint i=0;i<D;i++)m[i]=means[t*D+i];
  // R=G P G'+Q, right hand side=G P. Diagonal equilibration
  // changes only the linear-solve representation, not the covariance/model.
  mul(G,P,tmp);mult(tmp,G,R);for(uint i=0;i<144;i++)R[i]+=Q[i];
  for(uint i=0;i<D;i++){
   if(!isfinite(R[i*D+i])||R[i*D+i]<=0){failure[index]=10000+t;return;}
   scale[i]=sqrt(R[i*D+i]);
  }
  for(uint i=0;i<D;i++)for(uint j=0;j<D;j++){R[i*D+j]/=scale[i]*scale[j];tmp[i*D+j]/=scale[i];}
  // Pivoted LU elimination, simultaneously applying row operations to RHS.
  for(uint h=0;h<D;h++){
   uint pivot=h;float best=abs(R[h*D+h]);
   for(uint i=h+1;i<D;i++)if(abs(R[i*D+h])>best){best=abs(R[i*D+h]);pivot=i;}
   if(best==0||!isfinite(best)){failure[index]=20000+t;return;}
   if(pivot!=h)for(uint j=0;j<D;j++){
    float z=R[h*D+j];R[h*D+j]=R[pivot*D+j];R[pivot*D+j]=z;
    z=tmp[h*D+j];tmp[h*D+j]=tmp[pivot*D+j];tmp[pivot*D+j]=z;
   }
   for(uint i=h+1;i<D;i++){
    float a=R[i*D+h]/R[h*D+h];R[i*D+h]=0;
    for(uint j=h+1;j<D;j++)R[i*D+j]-=a*R[h*D+j];
    for(uint j=0;j<D;j++)tmp[i*D+j]-=a*tmp[h*D+j];
   }
  }
  for(int i=int(D)-1;i>=0;i--)for(uint j=0;j<D;j++){
   float z=tmp[i*D+j];for(uint h=i+1;h<D;h++)z-=R[i*D+h]*tmp[h*D+j];tmp[i*D+j]=z/R[i*D+i];
  }
  for(uint i=0;i<D;i++)for(uint j=0;j<D;j++){J[j*D+i]=tmp[i*D+j]/scale[i];gainhist[t*144+j*D+i]=J[j*D+i];}
  for(uint i=0;i<D;i++){float z=0;for(uint j=0;j<D;j++)z+=G[i*D+j]*m[j];v[i]=sm[i]-z;}
  for(uint i=0;i<D;i++){float z=m[i];for(uint j=0;j<D;j++)z+=J[i*D+j]*v[j];mp[i]=z;}
  for(uint i=0;i<D;i++)sm[i]=mp[i];
  // Stable Joseph RTS covariance: A P A'+J Q J'+J next J'.
  mul(J,G,A);for(uint i=0;i<D;i++)for(uint j=0;j<D;j++)A[i*D+j]=(i==j?1.0f:0.0f)-A[i*D+j];
  mul(A,P,tmp);mult(tmp,A,sc);
  mul(J,Q,tmp);mult(tmp,J,R);for(uint i=0;i<144;i++)sc[i]+=R[i];
  mul(J,next,tmp);mult(tmp,J,R);for(uint i=0;i<144;i++){sc[i]+=R[i];cov[t*144+i]=sc[i];}
  result[t*4]=sm[0];result[t*4+1]=sc[0];
 }
 float anchor_mean=result[anchor*4],anchor_var=result[anchor*4+1];
 for(uint t=0;t<times;t++)result[t*4+2]=result[t*4]-anchor_mean;
 result[anchor*4+3]=0;
 for(uint i=0;i<D;i++)v[i]=cov[anchor*144+i*D];
 for(int t=int(anchor)-1;t>=0;t--){
  for(uint i=0;i<D;i++){float z=0;for(uint j=0;j<D;j++)z+=gainhist[t*144+i*D+j]*v[j];mp[i]=z;}
  for(uint i=0;i<D;i++)v[i]=mp[i];
  result[t*4+3]=result[t*4+1]+anchor_var-2*v[0];
 }
 for(uint i=0;i<D;i++)v[i]=i==0?1.0f:0.0f;
 for(uint t=anchor+1;t<times;t++){
  for(uint i=0;i<D;i++){float z=0;for(uint j=0;j<D;j++)z+=gainhist[(t-1)*144+j*D+i]*v[j];mp[i]=z;}
  for(uint i=0;i<D;i++)v[i]=mp[i];
  float cross=0;for(uint i=0;i<D;i++)cross+=v[i]*cov[t*144+i*D];
  result[t*4+3]=result[t*4+1]+anchor_var-2*cross;
 }
 for(uint t=0;t<times;t++)for(uint j=0;j<4;j++)
  if(!isfinite(result[t*4+j])||((j==1||j==3)&&result[t*4+j]<0)){failure[index]=30000+t;return;}
}
'''


def smoother_source(model,nreg):
    d,ar=state_layout(model,nreg)
    from .api import _MODEL_SPECS
    spec=_MODEL_SPECS[model];c0=list(range(d));c1=list(range(d));c1[0]=1
    if spec['seasonal']:
        for i in (2,4):c0[i]=c0[i+1]=i;c1[i]=c1[i+1]=i+1
    if spec['ar2']:c0[ar]=c0[ar+1]=ar;c1[ar]=c1[ar+1]=ar+1
    constants=f"constant uint COL0[{d}]={{"+','.join(map(str,c0))+"};\n"+f"constant uint COL1[{d}]={{"+','.join(map(str,c1))+"};"
    source=SOURCE.replace('constant uint D=12;',f'constant uint D={d};\n'+constants)
    source=source.replace('144','(D*D)').replace('[12]','[D]').replace('+12)', '+D)')
    source=source.replace('uint c0[D],c1[D];','')
    source=source.replace('m[i]=0;c0[i]=i;c1[i]=i;if(i==0)c1[i]=1;', 'm[i]=0;')
    source=source.replace('if(i==2||i==3){c0[i]=2;c1[i]=3;}if(i==4||i==5){c0[i]=4;c1[i]=5;}', '')
    return source.replace('c0[','COL0[').replace('c1[','COL1[')


@lru_cache(maxsize=64)
def compile_smoother(model="vanilla_ar1",nreg=5):
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0' or os.environ.get('PYTORCH_MPS_FAST_MATH')!='0':
        raise ValueError('Explicit MPS fallback=0 and fast math=0 required')
    if not torch.backends.mps.is_available():raise RuntimeError('Native MPS unavailable')
    before=time.perf_counter();lib=torch.mps.compile_shader(smoother_source(model,nreg));kernel=lib.smooth
    return kernel,time.perf_counter()-before


@torch.no_grad()
def fixed_mps_trend_moments(p,u,anchor,*,workspace_bytes=1024**3,regressors=None):
    """Physical moments [draw,panel,time,mean/variance/change/change variance]."""
    target=StructuredMPSTarget(p);p=target.prepared
    if u.device.type!='mps' or u.dtype!=torch.float32 or u.ndim!=3 or u.shape[1:]!=(p.cells,len(p.parameter_names)) or not len(u):
        raise ValueError('MPS32 draw/panel/parameter array required')
    n=p.y.shape[1]
    if regressors is not None and (regressors.shape!=(len(u),p.cells,n,p.regressors.shape[-1]) or regressors.dtype!=u.dtype or regressors.device!=u.device):raise ValueError("Per-draw regressors must match draws/panels/time/regressors and dtype/device")
    if not 0<=anchor<n:raise ValueError('Valid anchor index required')
    d,_=state_layout(p.model,p.regressors.shape[-1]);history=2*d*d+d
    per_draw=p.cells*n*(history+4)*4
    tile=min(len(u),int(workspace_bytes)//per_draw)
    if tile<1:raise ValueError('Workspace too small to retain all panels for one draw')
    if tile*p.cells*n*history>=2**32:raise ValueError('Metal uint32 history limit exceeded')
    kernel,_=compile_smoother(p.model,p.regressors.shape[-1]);parts=[]
    for start in range(0,len(u),tile):
        z=u[start:start+tile]
        if regressors is None:
            _,G,Q,C,_,valid=model_components(z,*target.empty,p.model,S=p.priors['S'],sampling=p.sampling);F=target.F
        else:F,G,Q,C,_,valid=model_components(z,p.y,p.variance,p.observed,regressors[start:start+tile],p.model,S=p.priors['S'],sampling=p.sampling)
        if not bool(valid.all()):raise ValueError('Unsupported draw; no draw removal')
        count=len(z)*p.cells
        hist=torch.empty(count*n*history,device=u.device,dtype=u.dtype)
        out=torch.empty((len(z),p.cells,n,4),device=u.device,dtype=u.dtype)
        fail=torch.empty(count,device=u.device,dtype=torch.int32)
        kernel(p.y,p.variance,p.observed,F.contiguous(),G,Q,C,hist,out,fail,p.cells,n,count,int(anchor),int(F.ndim==4),threads=count,group_size=min(32,count))
        if bool((fail>=0).any()):raise FloatingPointError(f'GPU smoother failure codes {torch.unique(fail).cpu().tolist()}; no floor')
        if not bool(torch.isfinite(out).all()):raise FloatingPointError('Nonfinite GPU moments')
        parts.append(out)
    result=torch.cat(parts)
    scale=torch.as_tensor(p.scale,dtype=u.dtype,device=u.device)[None,:,None]
    offset=torch.as_tensor(p.offset,dtype=u.dtype,device=u.device)[None,:,None]
    result[:,:,:,0]=result[:,:,:,0]*scale+offset
    result[:,:,:,1]*=scale*scale;result[:,:,:,2]*=scale;result[:,:,:,3]*=scale*scale
    return result
