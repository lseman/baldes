//! Optional native preflight for two disposable resources and ordinary windows.
//! Input is snapshotted once per bucket; output preserves the original label order.
use super::*;

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Input {
    resources: [f64; 2],
    ng: u16,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Output {
    resources: [f64; 2],
    feasible: u8,
}
#[repr(C)]
#[derive(Default)]
struct Arc {
    head_lower: [f64; 2],
    head_upper: [f64; 2],
    tail_lower: [f64; 2],
    tail_upper: [f64; 2],
    consumption: [f64; 2],
    jump: [f64; 2],
    forbidden: u16,
    has_jump: u8,
}
extern "C" {
    fn baldes_extend_batch(
        arc: *const Arc,
        input: *const Input,
        output: *mut Output,
        count: usize,
        forward: bool,
    );
}

#[derive(Default)]
pub(super) struct ExtensionBatch {
    input: Vec<Input>,
    output: Vec<Output>,
}
impl ExtensionBatch {
    pub(super) fn supported<const D: usize>(g: &Graph<D>) -> bool {
        D == 2
            && g.resources.len() == 2
            && g.resources.iter().all(|r| *r == ResourceKind::Disposable)
            && !g.has_coupled_resources()
            && !g.has_time_windows()
    }

    pub(super) fn snapshot<const D: usize>(&mut self, storage: &LabelStorage<D>, labels: &[usize]) {
        self.input.clear();
        self.input.extend(labels.iter().map(|&i| Input {
            resources: [storage.hot[i].resources[0], storage.hot[i].resources[1]],
            ng: storage.hot[i].ng,
        }));
        self.output.resize(labels.len(), Output::default());
    }

    pub(super) fn run<const D: usize>(
        &mut self,
        ctx: Context<'_, D>,
        g: &Graph<D>,
        arc: &BucketArc,
        forward: bool,
    ) {
        let physical = &g.arcs[arc.physical];
        if ctx
            .forbidden_arcs
            .get(arc.physical)
            .copied()
            .unwrap_or(false)
            || physical.reduced_cost >= NO_ROUTE_COST
        {
            self.output.fill(Output::default());
            return;
        }
        let transition = if forward {
            &ctx.ng.forward
        } else {
            &ctx.ng.backward
        };
        let mut descriptor = Arc {
            forbidden: transition[arc.physical].forbidden,
            ..Default::default()
        };
        for r in 0..2 {
            let mut head = g.vertices[physical.head].bounds[r];
            if let Some(window) = physical.bounds[r] {
                head.lower = head.lower.max(window.lower);
                head.upper = head.upper.min(window.upper);
            }
            let mut tail = g.vertices[physical.tail].bounds[r];
            if physical.tail == g.source {
                if let Some(initial) = g.initial[r] {
                    tail.lower = tail.lower.max(initial);
                }
            }
            descriptor.head_lower[r] = head.lower;
            descriptor.head_upper[r] = head.upper;
            descriptor.tail_lower[r] = tail.lower;
            descriptor.tail_upper[r] = tail.upper;
            descriptor.consumption[r] = physical.consumption[r];
            descriptor.jump[r] = if forward {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            };
        }
        if let Some(jump) = arc.jump.filter(|_| ctx.phase.jump_arcs) {
            descriptor.has_jump = 1;
            let point = ctx.grid.direction(forward).corners[arc.from.vertex][jump];
            for (axis, &slot) in ctx.grid.resource_slots.iter().enumerate() {
                descriptor.jump[slot] = point[axis];
            }
        }
        // SAFETY: repr(C) descriptors and records match the native declarations;
        // both vectors contain exactly the snapshotted label count, are disjoint,
        // and remain allocated for this synchronous, non-throwing call.
        unsafe {
            baldes_extend_batch(
                &descriptor,
                self.input.as_ptr(),
                self.output.as_mut_ptr(),
                self.input.len(),
                forward,
            );
        }
    }

    pub(super) fn resources<const D: usize>(
        &self,
        position: usize,
    ) -> Option<crate::graph::ResourceValues<D>> {
        let output = self.output[position];
        (output.feasible != 0).then(|| crate::graph::ResourceValues::from_slice(&output.resources))
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/labeling/engine/extension_batch_tests.rs"]
mod tests;
