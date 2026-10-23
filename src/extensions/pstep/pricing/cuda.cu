// One block per endpoint/extreme job; lanes partition first-arc prefixes.
// Deterministic lexicographic reduction preserves CPU DFS ties. FP64, no FMA.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <climits>
struct Job { int start, end, maximum; };
struct Output { double cost, load, reduced, start_time, end_time; uint64_t extensions; int path[7], length, status; };
__device__ double dual(const double *d, int n, int row, int v) {
    return v == 0 || v == n+1 ? 0.0 : d[row*n+v-1];
}
template<bool Timed> __global__ void search(int n, int p, double capacity, const double *costs,
                       const double *demands, const double *duals, double weight,
                       const Job *jobs, int count, uint64_t limit, Output *output,
                       const double *ready, const double *due, const double *service) {
    int id = blockIdx.x;
    Job job = jobs[id];
    Output out = {}; out.reduced = INFINITY;
    int path[7], next[7], depth = 0;
    double load[7], cost[7], reduced[7], transit[7], barrier[7], latest[7];
    if (Timed) { transit[0]=0.0; barrier[0]=ready[job.start]; latest[0]=due[job.start]; }
    path[0] = job.start; next[0] = (p == 1 ? job.end : 1) + threadIdx.x;
    load[0] = job.start == 0 ? 0 : demands[job.start-1]; cost[0] = 0;
    double gi = dual(duals,n,2,job.start), gj = dual(duals,n,2,job.end);
    double price = job.maximum ? gi : gj;
    reduced[0] = -dual(duals,n,1,job.start) + dual(duals,n,1,job.end)
        + (job.maximum ? capacity*(gj-gi) : (gj-gi)*load[0]);
    while (depth >= 0) {
        int hops = depth+1;
        int end = hops == p ? job.end : n+1;
        int head = next[depth];
        next[depth] += depth == 0 ? blockDim.x : 1;
        if (head > end) { --depth; continue; }
        bool terminal = head == job.end, visited = false;
        for (int k=0; k<=depth; ++k) visited |= path[k] == head;
        if (visited || (head == n+1 && !terminal) || (path[depth] == 0 && head == n+1)
            || (terminal && job.start != 0 && hops != p) || (!terminal && hops >= p)) continue;
        if (++out.extensions > limit) { out.status = 1; break; }
        double arc = costs[size_t(path[depth])*(n+2)+head];
        if (isnan(arc)) continue;
        double demand = head <= n ? demands[head-1] : 0.0;
        double nl = load[depth]+demand;
        if (nl > capacity) continue;
        double nt=0.0, nb=0.0, nlst=0.0;
        if (Timed) {
            double duration=service[path[depth]]+arc;
            nt=transit[depth]+duration; nb=fmax(barrier[depth]+duration,ready[head]);
            nlst=fmin(latest[depth],due[head]-nt);
            if (!isfinite(nt) || !isfinite(nb)) { out.status=2; break; }
            if (nb>due[head] || nlst<ready[job.start]) continue;
        }
        double nc = cost[depth]+arc;
        double nr = reduced[depth]+weight*arc-dual(duals,n,0,path[depth])
            -dual(duals,n,0,head)+price*demand;
        if (!isfinite(nc) || !isfinite(nr)) { out.status = 2; break; }
        if (terminal) {
            double early=0.0, late=0.0;
            int modes=1;
            if (Timed) {
                early=fmin(fmax(nb-nt,ready[job.start]),nlst); late=nlst;
                if (job.start!=0 && head!=n+1) modes=2;
                if (job.start!=0 && head==n+1) early=late;
            }
            for (int mode=0; mode<modes; ++mode) {
                double start_time=mode==0 ? early : late;
                double end_time=Timed ? fmax(start_time+nt,nb) : 0.0;
                double candidate=Timed ? nr-dual(duals,n,3,job.start)*start_time
                    +dual(duals,n,3,head)*end_time : nr;
                if (!isfinite(candidate)) { out.status=2; break; }
                if (candidate<out.reduced) {
                    out.cost=nc; out.load=nl; out.reduced=candidate; out.length=depth+2;
                    out.start_time=start_time; out.end_time=end_time;
                    for (int k=0; k<=depth; ++k) out.path[k]=path[k];
                    out.path[depth+1]=head;
                }
            }
        } else {
            ++depth; path[depth]=head; load[depth]=nl; cost[depth]=nc; reduced[depth]=nr;
            next[depth] = depth+1 == p ? job.end : 1;
            if (Timed) { transit[depth]=nt; barrier[depth]=nb; latest[depth]=nlst; }
        }
    }
    __shared__ Output partial[128];
    partial[threadIdx.x] = out;
    __syncthreads();
    if (threadIdx.x == 0) {
        Output best = {}; best.reduced = INFINITY;
        uint64_t extensions = 0; int status = 0;
        for (int lane=0; lane<128; ++lane) {
            const Output &candidate = partial[lane];
            if (candidate.status) status = candidate.status;
            if (UINT64_MAX-extensions < candidate.extensions) status = 1;
            else extensions += candidate.extensions;
            bool earlier = false;
            if (candidate.length && candidate.reduced == best.reduced) {
                for (int k=0; k<candidate.length && k<best.length; ++k) {
                    if (candidate.path[k] != best.path[k]) {
                        earlier = candidate.path[k] < best.path[k]; break;
                    }
                }
            }
            if (candidate.length && (candidate.reduced < best.reduced || earlier)) best = candidate;
        }
        best.extensions = extensions;
        best.status = extensions > limit ? 1 : status;
        output[id] = best;
    }
}
// RAII releases all allocations even on a CUDA error. Errors never fall back
// silently to CPU or return a partial pricing certificate.
template<class T> struct Buffer {
    T *ptr = nullptr;
    ~Buffer() { if (ptr) cudaFree(ptr); }
    cudaError_t alloc(size_t count) { return cudaMalloc((void**)&ptr, count*sizeof(T)); }
    cudaError_t upload(const T *src, size_t count) {
        cudaError_t e = alloc(count);
        return e == cudaSuccess ? cudaMemcpy(ptr,src,count*sizeof(T),cudaMemcpyHostToDevice) : e;
    }
};
#define CHECK(expr) do { cudaError_t e = (expr); if (e != cudaSuccess) return (int)e; } while(0)
extern "C" int baldes_pstep_cuda(int n, int p, double capacity, const double *costs,
    const double *demands, const double *duals, double weight, const Job *jobs,
    int count, uint64_t limit, Output *output,
    const double *ready, const double *due, const double *service) {
    Buffer<double> c,d,u,a,b,s; Buffer<Job> j; Buffer<Output> o;
    CHECK(c.upload(costs,size_t(n+2)*(n+2))); CHECK(d.upload(demands,n));
    CHECK(u.upload(duals,(ready ? 4 : 3)*n)); CHECK(j.upload(jobs,count)); CHECK(o.alloc(count));
    if (ready) {
        CHECK(a.upload(ready,n+2)); CHECK(b.upload(due,n+2)); CHECK(s.upload(service,n+2));
        search<true><<<count,128>>>(n,p,capacity,c.ptr,d.ptr,u.ptr,weight,j.ptr,count,limit,o.ptr,a.ptr,b.ptr,s.ptr);
    } else {
        search<false><<<count,128>>>(n,p,capacity,c.ptr,d.ptr,u.ptr,weight,j.ptr,count,limit,o.ptr,nullptr,nullptr,nullptr);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(output,o.ptr,count*sizeof(Output),cudaMemcpyDeviceToHost));
    return 0;
}
