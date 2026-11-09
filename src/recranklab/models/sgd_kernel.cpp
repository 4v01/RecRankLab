// Original sequential explicit-feedback SGD. No races or fast-math reassociation.
#include <cstdint>
extern "C" double sgd_epoch(const int64_t* users,const int64_t* items,
 const float* rating,const int64_t* order,int64_t n,int d,float mean,
 float* p,float* q,float* bu,float* bi,float lr,float reg,float bias_reg) {
 double sse=0;
 for(int64_t t=0;t<n;++t){
  const int64_t row=order[t],u=users[row],i=items[row];
  float* pu=p+u*d;float* qi=q+i*d;
  float prediction=mean+bu[u]+bi[i];
  for(int k=0;k<d;++k)prediction+=pu[k]*qi[k];
  const float e=rating[row]-prediction;sse+=double(e)*e;
  bu[u]+=lr*(e-bias_reg*bu[u]);bi[i]+=lr*(e-bias_reg*bi[i]);
  for(int k=0;k<d;++k){const float old_p=pu[k],old_q=qi[k];
   pu[k]+=lr*(e*old_q-reg*old_p);qi[k]+=lr*(e*old_p-reg*old_q);}
 }
 return sse/n;
}
