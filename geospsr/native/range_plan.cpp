// Same stable-rank x/y partition as periodic_range.py. No kernel approximation.
#include <algorithm>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <limits>
struct Entry { int32_t id; uint8_t label; };
struct Plan {
    const double* x; int leaf; int64_t cap,total=0,maxfar=0;
    std::vector<int32_t> fi,di;
    std::vector<uint8_t> fl,dl,full;
    std::vector<int64_t> fo{0},doo{0};
    std::vector<double> anchors;
    Plan(const double* p,int n,int l,int64_t c):x(p),leaf(l),cap(c) {
        std::vector<int32_t> ids(n);std::iota(ids.begin(),ids.end(),0);
        std::stable_sort(ids.begin(),ids.end(),[&](int i,int j){return x[3*i]<x[3*j];});xs(ids);
    }
    void retain(std::vector<Entry> e,bool far,bool all=false,int sy=1) {
        if(total+(int64_t)e.size()>cap)throw std::runtime_error("native range plan entry cap exceeded");
        total+=e.size();
        if(far) {
            for(int dir=0;dir<2;++dir) {
                double sx=dir?-1.:1., s_y=sx*sy;
                double lx=std::numeric_limits<double>::infinity(),hx=-lx,ly=lx,hy=-lx;
                for(auto v:e)if(v.label==1-dir) {
                    double a=sx*x[3*v.id],b=s_y*x[3*v.id+1];
                    lx=std::min(lx,a);hx=std::max(hx,a);ly=std::min(ly,b);hy=std::max(hy,b);
                }
                anchors.insert(anchors.end(),{sx,s_y,lx,hx,ly,hy});
            }
            std::stable_sort(e.begin(),e.end(),[&](Entry a,Entry b){return x[3*a.id+2]<x[3*b.id+2];});
            for(auto a:e){fi.push_back(a.id);fl.push_back(a.label);}
            fo.push_back(fi.size());maxfar=std::max(maxfar,(int64_t)e.size());
        }else {
            for(auto a:e){di.push_back(a.id);dl.push_back(a.label);}
            doo.push_back(di.size());full.push_back(all);
        }
    }
    void ys(const std::vector<Entry>& e) {
        bool s=false,t=false;for(auto a:e){s|=a.label;t|=!a.label;}if(!s||!t)return;
        if((int)e.size()<=leaf){retain(e,false);return;}
        size_t mid=e.size()/2;
        std::vector<Entry> lo(e.begin(),e.begin()+mid),hi(e.begin()+mid,e.end());
        for(int sign:{1,-1}) {
            const auto& a=sign==1?lo:hi;const auto& b=sign==1?hi:lo;
            std::vector<Entry> rect;size_t ns=0;
            for(auto v:a)if(v.label){rect.push_back(v);++ns;}
            for(auto v:b)if(!v.label)rect.push_back(v);
            if(ns&&rect.size()>ns)retain(std::move(rect),true,false,sign);
        }
        ys(lo);ys(hi);
    }
    void xs(const std::vector<int32_t>& ids) {
        if((int)ids.size()<=leaf){std::vector<Entry> e;for(auto id:ids)e.push_back({id,1});retain(std::move(e),false,true);return;}
        size_t mid=ids.size()/2;std::vector<Entry> e; e.reserve(ids.size());
        for(size_t i=0;i<ids.size();++i)e.push_back({ids[i],(uint8_t)(i<mid)});
        std::stable_sort(e.begin(),e.end(),[&](Entry a,Entry b){return x[3*a.id+1]<x[3*b.id+1];});ys(e);
        xs(std::vector<int32_t>(ids.begin(),ids.begin()+mid));xs(std::vector<int32_t>(ids.begin()+mid,ids.end()));
    }
};
extern "C" void* range_build(const double* x,int n,int leaf,int64_t cap,char* error,int error_size) {
    try{return new Plan(x,n,leaf,cap);}catch(const std::exception& e){
        if(error_size>0){std::strncpy(error,e.what(),error_size-1);error[error_size-1]=0;}return nullptr;
    }
}
extern "C" void range_free(void* p){delete static_cast<Plan*>(p);}
extern "C" void range_counts(void* p,int64_t* v){auto& a=*static_cast<Plan*>(p);v[0]=a.fo.size()-1;v[1]=a.doo.size()-1;v[2]=a.fi.size();v[3]=a.di.size();v[4]=a.maxfar;}
extern "C" void* range_data(void* p,int field){auto& a=*static_cast<Plan*>(p);switch(field){
 case 0:return a.fi.data();case 1:return a.fl.data();case 2:return a.fo.data();case 3:return a.anchors.data();
 case 4:return a.di.data();case 5:return a.dl.data();case 6:return a.doo.data();case 7:return a.full.data();default:return nullptr;}}
