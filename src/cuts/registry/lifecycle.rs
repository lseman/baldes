use super::*;

impl CutRegistry {
    pub fn reset_separation_phase(&mut self) {
        self.separation_phase = 1;
    }

    pub fn increase_separation_phase(&mut self) {
        let phase = if self.separation_phase == 1 {
            3
        } else {
            self.separation_phase + 1
        };
        self.separation_phase = self.parameters.max_rank.min(phase);
    }

    /// or_rcsp_cut_separation_phase: (phase, at maximum).
    pub fn separation_phase(&self, options: &Options) -> Result<(i32, bool)> {
        let rows = options.get("BALDESrankOneCutsMaxNumRows")? as i32;
        let phase = self.separation_phase.min(rows);
        Ok((phase, phase >= rows))
    }

    pub fn increase_phase(&mut self, options: &Options) -> Result<i32> {
        self.parameters.max_rank = options.get("BALDESrankOneCutsMaxNumRows")? as i32;
        self.increase_separation_phase();
        Ok(self.separation_phase)
    }

    pub fn retire(&mut self, cut: usize) {
        if let Some(c) = self.cuts.get_mut(cut) {
            c.dual = 0.;
        }
        self.retired.insert(cut);
    }

    pub fn is_covering(&self, cut: usize) -> bool {
        self.cuts[cut].class == CutClass::Covering
    }
}
