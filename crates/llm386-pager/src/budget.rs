//! Per-section token budgets for the pager.
//!
//! `System` and `Task` are always fixed — they consume their actual
//! token cost from the model's input budget. The remainder
//! (the *variable* budget) is divided across the other sections by
//! their configured fraction. `Slack` is a reserved fraction that is
//! never filled and acts as anti-overflow headroom.
//!
//! ```text
//! input_budget
//!   - required_blocks_actual
//!   - task_tokens_actual
//!   - system_blocks_actual
//!   = variable_budget
//!
//! state_budget      = variable_budget * fractions[State]
//! plan_budget       = variable_budget * fractions[Plan]
//! recent_budget     = variable_budget * fractions[Recent]
//! retrieved_budget  = variable_budget * fractions[Retrieved]
//! tools_budget      = variable_budget * fractions[Tools]
//! background_budget = variable_budget * fractions[Background]
//! slack_reserved    = variable_budget * fractions[Slack]   (never filled)
//! ```
//!
//! Defaults split the variable budget across State, Plan, Recent,
//! Retrieved, Tools, Background, and a small Slack reserve that is
//! never filled. If the sum of fractions exceeds 1.0 they are
//! normalized down at allocation time so the per-section budgets
//! never sum above the variable budget.

use std::collections::BTreeMap;

use llm386_core::{SectionKind, TokenCount};

/// Per-section budget configuration. Cheap to clone.
#[derive(Clone, Debug)]
pub struct SectionBudgetTable {
    fractions: BTreeMap<SectionKind, f32>,
}

impl Default for SectionBudgetTable {
    /// Default split:
    ///
    /// State 0.10, Plan 0.05, Recent 0.20, Retrieved 0.40,
    /// Tools 0.15, Background 0.05, Slack 0.05  (sum 1.00)
    fn default() -> Self {
        let mut fractions = BTreeMap::new();
        fractions.insert(SectionKind::State, 0.10);
        fractions.insert(SectionKind::Plan, 0.05);
        fractions.insert(SectionKind::Recent, 0.20);
        fractions.insert(SectionKind::Retrieved, 0.40);
        fractions.insert(SectionKind::Tools, 0.15);
        fractions.insert(SectionKind::Background, 0.05);
        fractions.insert(SectionKind::Slack, 0.05);
        Self { fractions }
    }
}

impl SectionBudgetTable {
    /// Empty table — every variable section gets zero. Use as a base
    /// for a fully custom configuration.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            fractions: BTreeMap::new(),
        }
    }

    /// Override the fraction for a section. `fraction` is clamped to
    /// `[0.0, 1.0]`. Setting `System` or `Task` is silently ignored
    /// at allocation time (those sections are always fixed).
    pub fn set(&mut self, section: SectionKind, fraction: f32) {
        self.fractions.insert(section, fraction.clamp(0.0, 1.0));
    }

    /// Convenience builder form of [`set`].
    #[must_use]
    pub fn with(mut self, section: SectionKind, fraction: f32) -> Self {
        self.set(section, fraction);
        self
    }

    #[must_use]
    pub fn fraction(&self, section: SectionKind) -> f32 {
        self.fractions.get(&section).copied().unwrap_or(0.0)
    }

    /// Sum of fractions across non-fixed sections (System and Task
    /// are excluded since they are always fixed).
    #[must_use]
    pub fn variable_fractions_sum(&self) -> f32 {
        self.fractions
            .iter()
            .filter(|(s, _)| !matches!(s, SectionKind::System | SectionKind::Task))
            .map(|(_, f)| *f)
            .sum()
    }

    /// Compute concrete per-section budgets given the variable
    /// budget remaining after fixed sections (System, Task, required
    /// blocks) have taken their share.
    ///
    /// If the configured fractions sum above 1.0 they are normalized
    /// so the per-section allocations never exceed `variable`.
    #[must_use]
    pub fn allocate_variable(&self, variable: TokenCount) -> SectionAllocation {
        let sum = self.variable_fractions_sum();
        let scale = if sum > 1.0 { 1.0 / sum } else { 1.0 };

        let mut budgets = BTreeMap::new();
        for (&section, &frac) in &self.fractions {
            if matches!(section, SectionKind::System | SectionKind::Task) {
                continue;
            }
            #[allow(
                clippy::cast_precision_loss,
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss
            )]
            let n = (f64::from(variable.0) * f64::from(frac) * f64::from(scale)) as u32;
            budgets.insert(section, TokenCount(n));
        }
        SectionAllocation { budgets }
    }
}

/// Concrete per-section token budgets produced by
/// [`SectionBudgetTable::allocate_variable`].
#[derive(Clone, Debug, Default)]
pub struct SectionAllocation {
    budgets: BTreeMap<SectionKind, TokenCount>,
}

impl SectionAllocation {
    #[must_use]
    pub fn for_section(&self, section: SectionKind) -> TokenCount {
        self.budgets
            .get(&section)
            .copied()
            .unwrap_or(TokenCount::ZERO)
    }

    pub fn iter(&self) -> impl Iterator<Item = (SectionKind, TokenCount)> + '_ {
        self.budgets.iter().map(|(s, t)| (*s, *t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_fractions_sum_to_one() {
        let t = SectionBudgetTable::default();
        let sum = t.variable_fractions_sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum was {sum}");
    }

    #[test]
    fn allocate_variable_default_thousand() {
        let t = SectionBudgetTable::default();
        let a = t.allocate_variable(TokenCount(1_000));
        assert_eq!(a.for_section(SectionKind::State), TokenCount(100));
        assert_eq!(a.for_section(SectionKind::Plan), TokenCount(50));
        assert_eq!(a.for_section(SectionKind::Recent), TokenCount(200));
        assert_eq!(a.for_section(SectionKind::Retrieved), TokenCount(400));
        assert_eq!(a.for_section(SectionKind::Tools), TokenCount(150));
        assert_eq!(a.for_section(SectionKind::Background), TokenCount(50));
        assert_eq!(a.for_section(SectionKind::Slack), TokenCount(50));
    }

    #[test]
    fn allocate_normalizes_oversum() {
        // Two sections each at 0.75 → sum=1.5; should normalize to 0.5 each.
        let t = SectionBudgetTable::empty()
            .with(SectionKind::Recent, 0.75)
            .with(SectionKind::Retrieved, 0.75);
        let a = t.allocate_variable(TokenCount(1_000));
        assert_eq!(a.for_section(SectionKind::Recent), TokenCount(500));
        assert_eq!(a.for_section(SectionKind::Retrieved), TokenCount(500));
    }

    #[test]
    fn set_clamps_fraction_to_unit_range() {
        let mut t = SectionBudgetTable::empty();
        t.set(SectionKind::Recent, 5.0);
        assert!((t.fraction(SectionKind::Recent) - 1.0).abs() < f32::EPSILON);
        t.set(SectionKind::Recent, -1.0);
        assert!(t.fraction(SectionKind::Recent).abs() < f32::EPSILON);
    }

    #[test]
    fn system_and_task_fractions_are_ignored_at_allocation() {
        let t = SectionBudgetTable::empty()
            .with(SectionKind::System, 0.5)
            .with(SectionKind::Task, 0.5);
        let a = t.allocate_variable(TokenCount(1_000));
        // System / Task are fixed — no entries in the variable allocation.
        assert_eq!(a.for_section(SectionKind::System), TokenCount::ZERO);
        assert_eq!(a.for_section(SectionKind::Task), TokenCount::ZERO);
    }

    #[test]
    fn unset_section_yields_zero_fraction() {
        let t = SectionBudgetTable::empty();
        assert!(t.fraction(SectionKind::Recent).abs() < f32::EPSILON);
        let a = t.allocate_variable(TokenCount(1_000));
        assert_eq!(a.for_section(SectionKind::Recent), TokenCount::ZERO);
    }
}
