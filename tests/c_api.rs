use baldes::api::*;
use std::ffi::CStr;
use std::ptr;
#[test]
fn handles_own_routes_and_pricing_caps_are_per_call() {
    // SAFETY: Handles and all input/output buffers are created locally, kept
    // alive for each call, and freed once; no aliases mutate a live borrow.
    unsafe {
        let p=or_rcsp_new(4,0);assert!(!p.is_null());
        assert_eq!(or_rcsp_new_arc(p,0,1,1.),0);assert_eq!(or_rcsp_new_arc(p,1,3,1.),1);assert_eq!(or_rcsp_new_arc(p,0,3,3.),2);
        let mut cost=0.;let mut exact=0;let mut out=[ptr::null_mut();4];
        assert_eq!(or_rcsp_price_phase(p,0,out.as_mut_ptr(),1,&mut cost,&mut exact),1);assert_eq!(cost,2.);assert_eq!(exact,1);
        or_solution_delete(out[0]);
        assert_eq!(or_rcsp_price_phase(p,0,out.as_mut_ptr(),4,&mut cost,&mut exact),2);
        assert_eq!(or_solution_get_cost(out[0]),2.);assert_eq!(or_solution_get_cost(out[1]),3.);assert!(out[2].is_null());
        let mut n=0;let nodes=or_solution_get_nodes(out[0],&mut n);assert_eq!(std::slice::from_raw_parts(nodes,n as usize),[0,1,3]);
        or_rcsp_delete(p);
        assert_eq!(or_solution_get_cost(out[0]),2.);
        or_solution_delete(out[0]);or_solution_delete(out[1]);
    }
}
#[test]
fn invalid_calls_do_not_unwind_or_publish_partial_output() {
    // SAFETY: Only explicit null-pointer validation cases and valid local handles.
    unsafe {
        assert!(or_rcsp_new(-1,0).is_null());assert!(CStr::from_ptr(or_rcsp_last_error()).to_str().unwrap().contains("negative"));
        let p=or_rcsp_new(3,0);assert_eq!(or_rcsp_new_arc(p,0,99,0.),-1);assert_eq!(or_rcsp_new_arc(p,0,2,-1.),0);
        let mut cost=0.;let mut out=ptr::null_mut();assert_eq!(or_rcsp_solve(p,&mut cost,&mut out),1);or_solution_delete(out);
        assert_eq!(or_rcsp_solve(ptr::null_mut(),&mut cost,&mut out),-1);assert!(out.is_null());assert_eq!(cost,f64::INFINITY);
        or_rcsp_delete(p);
    }
}
#[test]
fn thread_local_errors_are_independent() {
    assert!(or_rcsp_new(-1,0).is_null());
    std::thread::spawn(|| {
        let p=or_rcsp_new(3,0);assert!(!p.is_null());
        // SAFETY: Locally created valid handle and thread-local error string.
        unsafe {assert_eq!(CStr::from_ptr(or_rcsp_last_error()).to_bytes(),b"");or_rcsp_delete(p);}
    }).join().unwrap();
    // SAFETY: Error string is thread-local and no subsequent call has changed it.
    unsafe {assert!(!CStr::from_ptr(or_rcsp_last_error()).to_bytes().is_empty());}
}
#[test]
fn pricing_failure_clears_outputs_and_recovers() {
    // SAFETY: Valid local handles, nul-terminated names and writable buffer slots.
    unsafe {
        let p=or_rcsp_new(2,0);or_rcsp_new_arc(p,0,1,-2.);
        let name=c"BALDEShardTimeThresholdInPricing";
        assert_eq!(or_rcsp_set_option(p,name.as_ptr(),-1.),0);
        let mut out=[ptr::null_mut();2];let mut cost=0.;let mut exact=1;
        assert_eq!(or_rcsp_price_phase(p,0,out.as_mut_ptr(),2,&mut cost,&mut exact),-1);
        assert!(out.iter().all(|p|p.is_null()));assert_eq!(exact,0);assert_eq!(cost,f64::INFINITY);
        assert_eq!(or_rcsp_set_option(p,name.as_ptr(),20.),0);
        assert_eq!(or_rcsp_price_phase(p,0,out.as_mut_ptr(),2,&mut cost,&mut exact),1);assert_eq!(cost,-2.);
        or_solution_delete(out[0]);or_rcsp_delete(p);
    }
}
