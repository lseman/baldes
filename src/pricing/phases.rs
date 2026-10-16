//! Column-generation pricing phase schedules.
use crate::{invalid, Result};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Phase {
    pub check_ng: bool,
    pub jump_arcs: bool,
    pub labels_per_bucket: i64,
}
impl Phase {
    pub fn exact(self) -> bool {
        self.check_ng && self.jump_arcs && self.labels_per_bucket == i64::MAX
    }
    pub fn validate(self) -> Result<()> {
        if self.labels_per_bucket < 1 {
            Err(invalid("positive phase label limit required"))
        } else {
            Ok(())
        }
    }
}
pub fn initialize(selector: i32, supplied: &[Phase]) -> Result<Vec<Phase>> {
    if !supplied.is_empty() {
        for p in supplied {
            p.validate()?;
        }
        return Ok(supplied.to_vec());
    }
    let p = |check_ng, jump_arcs, labels_per_bucket| Phase {
        check_ng,
        jump_arcs,
        labels_per_bucket,
    };
    let unlimited = i64::MAX;
    Ok(match selector {
        1 => vec![
            p(true, true, unlimited),
            p(true, false, 4),
            p(false, false, 1),
        ],
        2 => vec![p(true, false, 16), p(true, false, 4), p(false, false, 1)],
        3 => vec![p(true, false, 8), p(true, false, 4), p(false, false, 1)],
        4 => vec![
            p(true, true, unlimited),
            p(false, false, 8),
            p(false, false, 1),
        ],
        5 => vec![
            p(true, true, unlimited),
            p(true, false, unlimited),
            p(false, false, 8),
        ],
        _ => vec![
            p(true, true, unlimited),
            p(false, false, unlimited),
            p(false, false, 1),
        ],
    })
}
