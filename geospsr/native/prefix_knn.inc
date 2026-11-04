// Exact predecessor kNN for AFN's FSAI pattern. Coordinates are column-major.
// The search prunes by both bounding-box distance and minimum original index.
#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

template<class T> class PsrPrefixTree {
    struct Node {
        std::array<T, 3> low, high;
        int begin, end, left, right, minimum;
    };
    using Hit = std::pair<T, int>;
    const T* x;
    int stride;
    std::vector<int> indices;
    std::vector<Node> nodes;
    int build(int begin, int end) {
        Node node; node.begin=begin; node.end=end; node.left=node.right=-1;
        node.minimum=std::numeric_limits<int>::max();
        node.low.fill(std::numeric_limits<T>::infinity());
        node.high.fill(-std::numeric_limits<T>::infinity());
        for(int j=begin;j<end;++j) {
            int i=indices[j]; node.minimum=std::min(node.minimum,i);
            for(int d=0;d<3;++d) {
                T value=x[d*stride+i];
                node.low[d]=std::min(node.low[d],value);
                node.high[d]=std::max(node.high[d],value);
            }
        }
        int id=(int)nodes.size(); nodes.push_back(node);
        if(end-begin>32) {
            int axis=0;
            for(int d=1;d<3;++d) if(node.high[d]-node.low[d]>node.high[axis]-node.low[axis]) axis=d;
            int middle=(begin+end)/2;
            std::nth_element(indices.begin()+begin,indices.begin()+middle,indices.begin()+end,
                [&](int a,int b) {T u=x[axis*stride+a],v=x[axis*stride+b];return u<v || (u==v && a<b);});
            int left=build(begin,middle),right=build(middle,end);
            nodes[id].left=left; nodes[id].right=right;
        }
        return id;
    }
    T lower(int id,int q) const {
        T result=0;
        for(int d=0;d<3;++d) {
            T v=x[d*stride+q],delta=std::max(nodes[id].low[d]-v,std::max(v-nodes[id].high[d],T(0)));
            result+=delta*delta;
        }
        return result;
    }
    void search(int id,int q,int k,std::vector<Hit>& heap,T bound) const {
        const Node& node=nodes[id];
        if(node.minimum>=q) return;
        if((int)heap.size()==k && (bound>heap.front().first ||
            (bound==heap.front().first && node.minimum>=heap.front().second))) return;
        if(node.left<0) {
            for(int j=node.begin;j<node.end;++j) {
                int p=indices[j]; if(p>=q) continue;
                T distance=0;
                for(int d=0;d<3;++d) {T delta=x[d*stride+p]-x[d*stride+q];distance+=delta*delta;}
                Hit hit(distance,p);
                if((int)heap.size()<k) {heap.push_back(hit);std::push_heap(heap.begin(),heap.end());}
                else if(hit<heap.front()) {std::pop_heap(heap.begin(),heap.end());heap.back()=hit;std::push_heap(heap.begin(),heap.end());}
            }
        } else {
            int first=node.left,second=node.right;
            T a=lower(first,q),b=lower(second,q);
            if(b<a) {std::swap(first,second);std::swap(a,b);}
            search(first,q,k,heap,a);search(second,q,k,heap,b);
        }
    }
public:
    PsrPrefixTree(int n,const T* coordinates,int ldc): x(coordinates),stride(ldc),indices(n) {
        std::iota(indices.begin(),indices.end(),0);nodes.reserve(n/8+1);build(0,n);
    }
    void neighbors(int q,int k,std::vector<Hit>& heap,int* output) const {
        heap.clear();search(0,q,k,heap,T(0));
        std::sort_heap(heap.begin(),heap.end());
        for(int j=0;j<k;++j) output[j]=heap[j].second;
    }
};

template<class T> static void psr_prefix_knn(int k,int n,const T* x,int ldc,int* indices,int* counts) {
    PsrPrefixTree<T> tree(n,x,ldc);
    #pragma omp parallel
    {
        std::vector<std::pair<T,int>> heap;heap.reserve(k);
        #pragma omp for schedule(dynamic,64)
        for(int i=0;i<n;++i) {
            int* row=indices+i*k;
            int count=std::min(k,i+1);counts[i]=count;
            if(i<k) for(int j=0;j<=i;++j) row[j]=j;
            else {if(k>1) tree.neighbors(i,k-1,heap,row);row[k-1]=i;}
        }
    }
}
