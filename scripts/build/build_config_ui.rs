//! Terminal presentation for the standalone build configurator.
use super::Config;
use std::fmt::Write;
use std::path::Path;

const ACCENT: &str = "1;36";
const TEXT: &str = "0";
const MUTED: &str = "90";
const SELECTED: &str = "1;30;46";
const ENABLED: &str = "32";
const CHANGED: &str = "33";

pub(super) struct Item {
    pub id: usize,
    section: &'static str,
    title: &'static str,
    value: String,
    help: &'static str,
    enabled: Option<bool>,
}

impl Item {
    pub fn value(
        id: usize,
        section: &'static str,
        title: &'static str,
        value: impl Into<String>,
        help: &'static str,
    ) -> Self {
        Self {
            id,
            section,
            title,
            value: value.into(),
            help,
            enabled: None,
        }
    }

    pub fn toggle(
        id: usize,
        section: &'static str,
        title: &'static str,
        enabled: bool,
        help: &'static str,
    ) -> Self {
        Self {
            id,
            section,
            title,
            value: if enabled { "ON" } else { "OFF" }.into(),
            help,
            enabled: Some(enabled),
        }
    }

    pub fn editable(&self) -> bool {
        matches!(self.id, 8..=11 | 14)
    }
}

#[derive(Default)]
pub(super) struct State {
    pub selected: usize,
    pub scroll: usize,
}

pub(super) struct View<'a> {
    pub config: &'a Config,
    pub original: &'a Config,
    pub path: &'a Path,
    pub message: &'a str,
    pub editing: Option<(&'a str, &'a str)>,
}

struct Frame {
    output: String,
    color: bool,
    width: usize,
    height: usize,
}

impl Frame {
    fn new(color: bool, width: usize, height: usize) -> Self {
        Self {
            output: String::from("\x1b[H\x1b[2J"),
            color,
            width,
            height,
        }
    }

    fn put(&mut self, y: usize, x: usize, width: usize, text: &str, style: &str) {
        if y == 0 || y > self.height || x == 0 || x > self.width {
            return;
        }
        let width = width.min(self.width + 1 - x);
        let text = fit(text, width);
        let _ = write!(self.output, "\x1b[{y};{x}H");
        if self.color {
            let _ = write!(self.output, "\x1b[{style}m");
        }
        self.output.push_str(&text);
        self.output
            .extend(std::iter::repeat(' ').take(width.saturating_sub(cells(&text))));
        if self.color {
            self.output.push_str("\x1b[0m");
        }
    }

    fn panel(&mut self, y: usize, x: usize, width: usize, height: usize, title: &str) {
        self.put(y, x, width, &format!("╭{}╮", "─".repeat(width - 2)), MUTED);
        self.put(y, x + 2, title.len() + 2, &format!(" {title} "), ACCENT);
        for row in y + 1..y + height - 1 {
            self.put(row, x, 1, "│", MUTED);
            self.put(row, x + width - 1, 1, "│", MUTED);
        }
        self.put(
            y + height - 1,
            x,
            width,
            &format!("╰{}╯", "─".repeat(width - 2)),
            MUTED,
        );
    }
}

// Common terminal cell widths, including combining marks and CJK paths. Never
// interpolate configuration text into ANSI control sequences.
fn cell_width(c: char) -> usize {
    match c as u32 {
        0x0300..=0x036f
        | 0x1ab0..=0x1aff
        | 0x1dc0..=0x1dff
        | 0x200b..=0x200f
        | 0x202a..=0x202e
        | 0x2060..=0x206f
        | 0x20d0..=0x20ff
        | 0xfe00..=0xfe0f
        | 0xfe20..=0xfe2f => 0,
        0x1100..=0x115f
        | 0x2329..=0x232a
        | 0x2e80..=0xa4cf
        | 0xac00..=0xd7a3
        | 0xf900..=0xfaff
        | 0xfe10..=0xfe19
        | 0xfe30..=0xfe6f
        | 0xff01..=0xff60
        | 0xffe0..=0xffe6
        | 0x1f300..=0x1faff
        | 0x20000..=0x3fffd => 2,
        _ => 1,
    }
}

fn cells(text: &str) -> usize {
    text.chars()
        .filter(|c| !c.is_control())
        .map(cell_width)
        .sum()
}

fn fit(text: &str, width: usize) -> String {
    let safe: String = text
        .chars()
        .filter(|c| !c.is_control() && cell_width(*c) > 0)
        .collect();
    if cells(&safe) <= width {
        return safe;
    }
    let mut result = String::new();
    let mut used = 0;
    for c in safe.chars() {
        let next = cell_width(c);
        if used + next > width.saturating_sub(1) {
            break;
        }
        result.push(c);
        used += next;
    }
    if width > 0 {
        result.push('…');
    }
    result
}

fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && cells(&line) + 1 + cells(word) > width {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(&fit(word, width));
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

// Includes section headings without allowing the cursor to land on one.
fn list_lines(items: &[Item]) -> Vec<Option<usize>> {
    let mut lines = Vec::new();
    let mut section = "";
    for (i, item) in items.iter().enumerate() {
        if section != item.section {
            lines.push(None);
            section = item.section;
        }
        lines.push(Some(i));
    }
    lines
}

fn scroll_to_selection(
    lines: &[Option<usize>],
    selected: usize,
    height: usize,
    scroll: usize,
) -> usize {
    let pos = lines
        .iter()
        .position(|&line| line == Some(selected))
        .unwrap_or(0);
    let mut start = scroll.min(lines.len().saturating_sub(height));
    if pos < start {
        start = pos.saturating_sub(1);
    }
    if pos >= start + height {
        start = pos + 1 - height;
    }
    start
}

pub(super) fn render(
    width: usize,
    height: usize,
    view: &View<'_>,
    state: &mut State,
    color: bool,
) -> String {
    let mut frame = Frame::new(color, width, height);
    if width < 64 || height < 20 {
        frame.put(1, 1, width, "BALDES · Build configuration", ACCENT);
        frame.put(
            3,
            1,
            width,
            "Resize to at least 64 columns × 20 rows.",
            TEXT,
        );
        frame.put(
            5,
            1,
            width,
            "Your changes are kept while you resize. Q cancels.",
            MUTED,
        );
        return frame.output;
    }
    let items = view.config.rows();
    let item = &items[state.selected];
    let dirty = view.config != view.original;
    let area = width - 4;
    let wide = width >= 104;
    let left = if wide { area * 3 / 5 } else { area };
    let panel_height = height - 8;
    let bottom = 5 + panel_height - 1;
    let list_height = panel_height - if wide { 3 } else { 6 };
    let inner = left - 4;

    frame.put(2, 3, 8, "BALDES", ACCENT);
    frame.put(2, 12, area - 9, "BUILD CONFIGURATION", "1");
    frame.put(
        3,
        3,
        area.saturating_sub(20),
        &format!(
            "{}  /  {}",
            view.path.display(),
            if view.config.release {
                "release"
            } else {
                "development"
            }
        ),
        MUTED,
    );
    frame.put(
        3,
        width - 20,
        18,
        if dirty {
            "● Unsaved changes"
        } else {
            "○ Settings loaded"
        },
        if dirty { CHANGED } else { MUTED },
    );

    frame.panel(5, 3, left, panel_height, "BUILD OPTIONS");
    frame.put(
        5,
        3 + left - 10,
        7,
        &format!(" {:02}/{:02} ", state.selected + 1, items.len()),
        MUTED,
    );
    let lines = list_lines(&items);
    state.scroll = scroll_to_selection(&lines, state.selected, list_height, state.scroll);
    for (row, entry) in lines
        .iter()
        .enumerate()
        .skip(state.scroll)
        .take(list_height)
    {
        let y = 6 + row - state.scroll;
        match *entry {
            None => {
                let section = items[lines[row + 1].unwrap()].section;
                frame.put(y, 5, inner, section, ACCENT);
            }
            Some(index) => {
                let entry = &items[index];
                let selected = index == state.selected;
                let value_width = 14.min(inner / 3);
                let label_width = inner - value_width - 4;
                let style = if selected { SELECTED } else { TEXT };
                frame.put(y, 5, inner, "", style);
                frame.put(y, 5, 2, if selected { "›" } else { "" }, style);
                frame.put(y, 7, label_width, entry.title, style);
                let value = fit(&entry.value, value_width - 2);
                let value = format!(
                    "[{:>pad$}]",
                    value,
                    pad = value_width - 2 - cells(&value) + value.chars().count()
                );
                let value_style = if selected {
                    SELECTED
                } else if entry.enabled == Some(true) {
                    ENABLED
                } else if entry.enabled == Some(false) {
                    MUTED
                } else {
                    TEXT
                };
                frame.put(y, 5 + inner - value_width, value_width, &value, value_style);
            }
        }
    }
    if state.scroll + list_height < lines.len() {
        frame.put(bottom, 3 + left - 20, 16, " ↓ More options ", MUTED);
    }
    if wide {
        let x = 3 + left + 2;
        let right = area - left - 2;
        frame.panel(5, x, right, panel_height, "DETAILS");
        let inner = right - 4;
        frame.put(7, x + 2, inner, item.title, "1");
        let help = wrap(item.help, inner);
        let max_help = panel_height.saturating_sub(11);
        for (i, line) in help.iter().take(max_help).enumerate() {
            frame.put(9 + i, x + 2, inner, line, TEXT);
        }
        let y = bottom - 6;
        frame.put(
            y,
            x + 2,
            inner,
            if item.editable() {
                "ENTER  Edit value"
            } else {
                "SPACE / ENTER  Change"
            },
            ACCENT,
        );
        frame.put(y + 2, x + 2, inner, "BUILD SUMMARY", MUTED);
        frame.put(
            y + 3,
            x + 2,
            inner,
            if view.config.cli {
                "Library + command-line program"
            } else {
                "Library"
            },
            TEXT,
        );
        let backends = [
            (view.config.clp, "Clp"),
            (view.config.highs, "HiGHS"),
            (view.config.gurobi, "Gurobi"),
        ]
        .into_iter()
        .filter_map(|(enabled, name)| enabled.then_some(name))
        .collect::<Vec<_>>()
        .join(" + ");
        let backends = if backends.is_empty() {
            "No optional LP backends"
        } else {
            &backends
        };
        frame.put(y + 4, x + 2, inner, backends, TEXT);
        frame.put(
            y + 5,
            x + 2,
            inner,
            "Native IPM, pricing, HGS always included",
            MUTED,
        );
    } else {
        frame.put(bottom - 3, 5, inner, &"─".repeat(inner), MUTED);
        for (i, line) in wrap(item.help, inner).iter().take(2).enumerate() {
            frame.put(bottom - 2 + i, 5, inner, line, MUTED);
        }
    }
    frame.put(
        height - 2,
        3,
        area,
        "↑↓ / j k Move   Space Change   s Save & exit   q Cancel",
        TEXT,
    );
    let status = if view.message.is_empty() {
        "Home / End · PgUp / PgDn · Enter edits text fields"
    } else {
        view.message
    };
    frame.put(
        height - 1,
        3,
        area,
        status,
        if view.message.is_empty() {
            MUTED
        } else {
            CHANGED
        },
    );

    if let Some((prompt, value)) = view.editing {
        let w = area.min(76);
        let x = (width - w) / 2 + 1;
        let y = (height - 9) / 2 + 1;
        for row in y..y + 9 {
            frame.put(row, x, w, "", TEXT);
        }
        frame.panel(y, x, w, 9, "EDIT VALUE");
        frame.put(y + 2, x + 3, w - 6, prompt, "1");
        // Keep the end of a long field visible while typing.
        let mut tail = value;
        while cells(tail) > w - 9 {
            tail = &tail[tail.chars().next().unwrap().len_utf8()..];
        }
        frame.put(y + 4, x + 3, w - 6, &format!("{tail}▏"), SELECTED);
        frame.put(
            y + 6,
            x + 3,
            w - 6,
            "Enter Apply   Esc Back   Ctrl-U Clear",
            MUTED,
        );
    }
    frame.output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrolling_keeps_every_setting_visible_after_resizing() {
        let rows = Config::default().rows();
        let lines = list_lines(&rows);
        let mut scroll = 0;
        for height in [20, 6, 10, 1, 12] {
            for selected in (0..rows.len()).chain((0..rows.len()).rev()) {
                scroll = scroll_to_selection(&lines, selected, height, scroll);
                assert!(lines[scroll..lines.len().min(scroll + height)].contains(&Some(selected)));
            }
        }
        let mut ids: Vec<_> = rows.iter().map(|item| item.id).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..17).collect::<Vec<_>>());
    }

    #[test]
    fn monochrome_and_editing_survive_small_and_wide_terminals() {
        let config = Config::default();
        let view = View {
            config: &config,
            original: &config,
            path: Path::new(".config"),
            message: "",
            editing: Some(("Target triple", "aarch64-unknown-linux-gnu")),
        };
        for (width, height) in [
            (0, 0),
            (1, 1),
            (63, 19),
            (64, 20),
            (80, 24),
            (104, 24),
            (120, 32),
        ] {
            let text = render(width, height, &view, &mut State::default(), false);
            assert!(!text.contains("\x1b[1;36m"));
            if width >= 64 && height >= 20 {
                assert!(text.contains("EDIT VALUE"));
                assert!(text.contains("Build profile"));
            }
        }
        let text = render(120, 32, &view, &mut State::default(), true);
        assert!(text.contains("\x1b[1;36m"));
    }

    #[test]
    fn long_values_and_control_sequences_do_not_overrun_fields() {
        assert_eq!(cells(&fit("/opt/库/HiGHS", 8)), 8);
        assert_eq!(fit("abcdef", 3), "ab…");
        assert_eq!(fit("abc", 0), "");
        assert!(!fit("unsafe\x1b[31m\npath", 40).contains('\x1b'));
        let mut frame = Frame::new(true, 64, 20);
        frame.put(21, 1, 64, "outside", TEXT);
        frame.put(1, 65, 1, "outside", TEXT);
        assert!(!frame.output.contains("outside"));
    }
}
