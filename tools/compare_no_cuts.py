"""Differential smoke suite for the active Rust C ABI and validated C++ C ABI.

Run after `cargo build --release`, from the BALDES root:
    python tools/compare_no_cuts.py --cases 80
Uses ctypes only; no Rust dependencies, Python package install or C++ rebuild.
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]

def load(path):
    lib=C.CDLL(str(path),mode=C.RTLD_LOCAL)
    p=C.c_void_p;i=C.c_int;d=C.c_double
    declarations={
        'or_rcsp_new':(p,[i,i]),'or_rcsp_delete':(None,[p]),
        'or_rcsp_new_arc':(i,[p,i,i,d]),'or_rcsp_set_arc_consumption':(None,[p,i,i,d,d]),
        'or_rcsp_set_vertex_bounds':(i,[p,i,i,d,d]),
        'or_rcsp_set_elementary':(i,[p,i,C.c_size_t]),
        'or_rcsp_solve':(i,[p,C.POINTER(d),C.POINTER(p)]),
        'or_solution_get_arc_ids':(C.POINTER(i),[p,C.POINTER(i)]),
        'or_solution_get_cost':(d,[p]),'or_solution_delete':(None,[p]),
        'or_rcsp_last_error':(C.c_char_p,[]),
    }
    for name,(res,args) in declarations.items():
        f=getattr(lib,name);f.restype=res;f.argtypes=args
    return lib

def solve(lib,arcs,windows,elementary):
    p=lib.or_rcsp_new(len(windows),1)
    assert p,lib.or_rcsp_last_error()
    out=C.c_void_p()
    try:
        for tail,head,cost,use in arcs:
            a=lib.or_rcsp_new_arc(p,tail,head,cost);assert a>=0
            lib.or_rcsp_set_arc_consumption(p,a,0,use,use)
        for v,(lower,upper) in enumerate(windows):
            assert lib.or_rcsp_set_vertex_bounds(p,v,0,lower,upper)==0
        assert lib.or_rcsp_set_elementary(p,int(elementary),0 if elementary else C.c_size_t(-1).value)==0
        cost=C.c_double();status=lib.or_rcsp_solve(p,C.byref(cost),C.byref(out))
        assert status>=0,lib.or_rcsp_last_error()
        ids=[]
        if status:
            n=C.c_int();buf=lib.or_solution_get_arc_ids(out,C.byref(n));ids=list(buf[:n.value])
        return status,cost.value,ids
    finally:
        if out:lib.or_solution_delete(out)
        lib.or_rcsp_delete(p)

def replay(result,arcs,windows,elementary):
    status,cost,ids=result
    if not status:return
    vertex=0;q=windows[0][0];total=0.;seen={0}
    for id in ids:
        tail,head,value,use=arcs[id]
        assert vertex==tail
        assert not elementary or head not in seen
        q=max(q+use,windows[head][0]);assert q<=windows[head][1]
        total+=value;seen.add(head);vertex=head
    assert vertex==len(windows)-1 and abs(total-cost)<1e-9

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases',type=int,default=80)
    ap.add_argument('--cpp',type=Path,default=Path(os.environ.get('OPENRSCP_ROOT',ROOT.parent/'openrscp'))/'python/src/openrscp/_core.cpython-314-x86_64-linux-gnu.so')
    ap.add_argument('--rust',type=Path,default=ROOT/'target/release/libbaldes.so')
    ap.add_argument('--json',type=Path)
    args=ap.parse_args()
    rust,cpp=load(args.rust),load(args.cpp)
    rng=random.Random(201)
    records=[]
    for case in range(args.cases):
        elementary=bool(case%2)
        windows=[(0.,5.)]+[(float(rng.randrange(3)),5.) for _ in range(5)]
        arcs=[]
        for tail in range(5):
            for head in range(1,6):
                if head==tail or (not elementary and head<=tail) or rng.randrange(3)==0:continue
                arcs.append((tail,head,rng.randrange(-12,13)/4.,1.))
        a=solve(rust,arcs,windows,elementary);b=solve(cpp,arcs,windows,elementary)
        assert a[0]==b[0] and (not a[0] or abs(a[1]-b[1])<1e-8),(case,a,b,arcs,windows)
        replay(a,arcs,windows,elementary);replay(b,arcs,windows,elementary)
        records.append({'case':case,'elementary':elementary,'feasible':bool(a[0]),'rust_cost':a[1] if a[0] else None,'cpp_cost':b[1] if b[0] else None,'same_physical_route':a[2]==b[2]})
    report={'cases':len(records),'cost_and_feasibility_match':True,'all_routes_replayed':True,'records':records}
    if args.json:args.json.parent.mkdir(parents=True,exist_ok=True);args.json.write_text(json.dumps(report,indent=2)+'\n')
    print(f'{len(records)}/{len(records)} no-cut cases match costs/feasibility; all routes replayed')
if __name__=='__main__':main()
