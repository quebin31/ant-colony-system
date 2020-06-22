use ndarray::Array2;
use prettytable::{cell, format::consts::FORMAT_BOX_CHARS, Row, Table};

pub trait ToCharIndex {
    fn to_char_index(&self) -> char;
}

impl ToCharIndex for usize {
    fn to_char_index(&self) -> char {
        let offset = *self as u8;
        (b'A' + offset) as char
    }
}

pub trait ToDisplayPath {
    fn to_display_path(&self) -> Result<String, std::fmt::Error>;
}

impl<T: ToCharIndex> ToDisplayPath for Vec<T> {
    fn to_display_path(&self) -> Result<String, std::fmt::Error> {
        use std::fmt::Write;

        let mut out = String::new();
        write!(out, "[")?;
        for i in self.iter().take(self.len() - 1) {
            write!(out, "{}, ", i.to_char_index())?;
        }

        if let Some(last) = self.last() {
            write!(out, "{}", last.to_char_index())?;
        }

        write!(out, "]")?;
        Ok(out)
    }
}

pub fn pretty_matrix(matrix: &Array2<f64>, digits: usize) -> Table {
    let mut table = Table::new();
    table.set_format(*FORMAT_BOX_CHARS);

    let mut titles: Row = (0..matrix.shape()[1]).map(|v| v.to_char_index()).into();
    titles.insert_cell(0, cell![""]);
    table.set_titles(titles);

    for r in 0..(matrix.shape()[0]) {
        let mut row: Row = matrix
            .row(r)
            .iter()
            .map(|v| format!("{1:.0$}", digits, v))
            .into();

        row.insert_cell(0, cell![r.to_char_index()]);
        table.add_row(row);
    }

    table
}
