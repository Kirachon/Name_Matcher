use std::fs;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use chrono::{DateTime, Utc};
use rust_xlsxwriter::{Color, Format, FormatAlign, Workbook, Worksheet};

use crate::matching::MatchPair;

#[derive(Debug, Clone)]
pub struct SummaryContext {
    pub db_name: String,
    pub table1: String,
    pub table2: String,

    pub total_table1: usize,
    pub total_table2: usize,

    pub matches_algo1: usize,
    pub matches_algo2: usize,
    pub matches_fuzzy: usize,

    pub overlap_count: usize,
    pub unique_algo1: usize,
    pub unique_algo2: usize,

    pub fetch_time: Duration,
    pub match1_time: Duration,
    pub match2_time: Duration,
    pub export_time: Duration,

    pub mem_used_start_mb: u64,
    pub mem_used_end_mb: u64,

    pub started_utc: DateTime<Utc>,
    pub ended_utc: DateTime<Utc>,
    pub duration_secs: f64,

    pub algo_used: String,
    pub gpu_used: bool,
    pub gpu_total_mb: u64,
    pub gpu_free_mb_end: u64,
}

fn ensure_parent_dir(path: &str) -> Result<()> {
    let p = Path::new(path);
    if let Some(parent) = p.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn header_format() -> Format {
    Format::new().set_bold().set_align(FormatAlign::Center)
}

fn row_format_even() -> Format {
    Format::new().set_background_color(Color::RGB(0xF2F2F2))
}

fn write_algo1_sheet(ws: &mut Worksheet, matches: &[MatchPair]) -> Result<()> {
    let headers = [
        "Table1_ID","Table1_UUID","Table1_FirstName","Table1_LastName","Table1_Birthdate",
        "Table2_ID","Table2_UUID","Table2_FirstName","Table2_LastName","Table2_Birthdate",
        "is_matched_Infnbd","Confidence","MatchedFields",
    ];
    let hfmt = header_format();
    for (c, h) in headers.iter().enumerate() {
        ws.write_string_with_format(0, c as u16, *h, &hfmt)?;
    }

    let even = row_format_even();
    for (i, m) in matches.iter().enumerate() {
        let r = (i + 1) as u32;
        if i % 2 == 0 { ws.set_row_format(r, &even)?; }
        ws.write_number(r, 0, m.person1.id as f64)?;
        ws.write_string(r, 1, m.person1.uuid.as_deref().unwrap_or(""))?;
        ws.write_string(r, 2, m.person1.first_name.as_deref().unwrap_or(""))?;
        ws.write_string(r, 3, m.person1.last_name.as_deref().unwrap_or(""))?;
        ws.write_string(r, 4, &m.person1.birthdate.as_ref().map(|d| d.to_string()).unwrap_or_default())?;
        ws.write_number(r, 5, m.person2.id as f64)?;
        ws.write_string(r, 6, m.person2.uuid.as_deref().unwrap_or(""))?;
        ws.write_string(r, 7, m.person2.first_name.as_deref().unwrap_or(""))?;
        ws.write_string(r, 8, m.person2.last_name.as_deref().unwrap_or(""))?;
        ws.write_string(r, 9, &m.person2.birthdate.as_ref().map(|d| d.to_string()).unwrap_or_default())?;
        ws.write_string(r,10, if m.is_matched_infnbd {"true"} else {"false"})?;
        ws.write_number(r,11, m.confidence as f64)?;
        ws.write_string(r,12, &m.matched_fields.join(";"))?;
    }
    Ok(())
}

fn write_algo2_sheet(ws: &mut Worksheet, matches: &[MatchPair]) -> Result<()> {
    let headers = [
        "Table1_ID","Table1_UUID","Table1_FirstName","Table1_MiddleName","Table1_LastName","Table1_Birthdate",
        "Table2_ID","Table2_UUID","Table2_FirstName","Table2_MiddleName","Table2_LastName","Table2_Birthdate",
        "is_matched_Infnmnbd","Confidence","MatchedFields",
    ];
    let hfmt = header_format();
    for (c, h) in headers.iter().enumerate() {
        ws.write_string_with_format(0, c as u16, *h, &hfmt)?;
    }

    let even = row_format_even();
    for (i, m) in matches.iter().enumerate() {
        let r = (i + 1) as u32;
        if i % 2 == 0 { ws.set_row_format(r, &even)?; }
        ws.write_number(r, 0, m.person1.id as f64)?;
        ws.write_string(r, 1, m.person1.uuid.as_deref().unwrap_or(""))?;
        ws.write_string(r, 2, m.person1.first_name.as_deref().unwrap_or(""))?;
        ws.write_string(r, 3, m.person1.middle_name.as_deref().unwrap_or(""))?;
        ws.write_string(r, 4, m.person1.last_name.as_deref().unwrap_or(""))?;
        ws.write_string(r, 5, &m.person1.birthdate.as_ref().map(|d| d.to_string()).unwrap_or_default())?;
        ws.write_number(r, 6, m.person2.id as f64)?;
        ws.write_string(r, 7, m.person2.uuid.as_deref().unwrap_or(""))?;
        ws.write_string(r, 8, m.person2.first_name.as_deref().unwrap_or(""))?;
        ws.write_string(r, 9, m.person2.middle_name.as_deref().unwrap_or(""))?;
        ws.write_string(r,10, m.person2.last_name.as_deref().unwrap_or(""))?;
        ws.write_string(r,11, &m.person2.birthdate.as_ref().map(|d| d.to_string()).unwrap_or_default())?;
        ws.write_string(r,12, if m.is_matched_infnmnbd {"true"} else {"false"})?;
        ws.write_number(r,13, m.confidence as f64)?;
        ws.write_string(r,14, &m.matched_fields.join(";"))?;
    }
    Ok(())
}

fn write_summary_sheet(ws: &mut Worksheet, ctx: &SummaryContext) -> Result<()> {
    let hfmt = header_format();
    let mut row: u32 = 0;

    ws.write_string_with_format(row, 0, "Summary", &hfmt)?; row += 2;

    let kv = |ws: &mut Worksheet, r: &mut u32, k: &str, v: &str| -> Result<()> {
        ws.write_string(*r, 0, k)?;
        ws.write_string(*r, 1, v)?;
        *r += 1;
        Ok(())
    };

    kv(ws, &mut row, "Database", &ctx.db_name)?;
    kv(ws, &mut row, "Table 1", &ctx.table1)?;
    kv(ws, &mut row, "Table 2", &ctx.table2)?;

    kv(ws, &mut row, "Total records (Table1)", &ctx.total_table1.to_string())?;
    kv(ws, &mut row, "Total records (Table2)", &ctx.total_table2.to_string())?;

    kv(ws, &mut row, "Matches (Algorithm 1)", &ctx.matches_algo1.to_string())?;
    kv(ws, &mut row, "Matches (Algorithm 2)", &ctx.matches_algo2.to_string())?;
    kv(ws, &mut row, "Matches (Fuzzy)", &ctx.matches_fuzzy.to_string())?;
    kv(ws, &mut row, "Overlap (A1 ∩ A2)", &ctx.overlap_count.to_string())?;
    kv(ws, &mut row, "Unique to A1", &ctx.unique_algo1.to_string())?;
    kv(ws, &mut row, "Unique to A2", &ctx.unique_algo2.to_string())?;

    // Derived stats
    let u1_t1 = ctx.total_table1.saturating_sub(ctx.matches_algo1);
    let u1_t2 = ctx.total_table2.saturating_sub(ctx.matches_algo1);
    let u2_t1 = ctx.total_table1.saturating_sub(ctx.matches_algo2);
    let u2_t2 = ctx.total_table2.saturating_sub(ctx.matches_algo2);
    let rate = |num: usize, den: usize| if den == 0 { "0.000".to_string() } else { format!("{:.3}", (num as f64)/(den as f64)) };
    kv(ws, &mut row, "Unmatched by A1 (Table1)", &u1_t1.to_string())?;
    kv(ws, &mut row, "Unmatched by A1 (Table1) rate", &rate(u1_t1, ctx.total_table1))?;
    kv(ws, &mut row, "Unmatched by A1 (Table2)", &u1_t2.to_string())?;
    kv(ws, &mut row, "Unmatched by A1 (Table2) rate", &rate(u1_t2, ctx.total_table2))?;
    kv(ws, &mut row, "Unmatched by A2 (Table1)", &u2_t1.to_string())?;
    kv(ws, &mut row, "Unmatched by A2 (Table1) rate", &rate(u2_t1, ctx.total_table1))?;
    kv(ws, &mut row, "Unmatched by A2 (Table2)", &u2_t2.to_string())?;
    kv(ws, &mut row, "Unmatched by A2 (Table2) rate", &rate(u2_t2, ctx.total_table2))?;

    // Fuzzy unmatched
    let uf_t1 = ctx.total_table1.saturating_sub(ctx.matches_fuzzy);
    let uf_t2 = ctx.total_table2.saturating_sub(ctx.matches_fuzzy);
    kv(ws, &mut row, "Unmatched by Fuzzy (Table1)", &uf_t1.to_string())?;
    kv(ws, &mut row, "Unmatched by Fuzzy (Table2)", &uf_t2.to_string())?;

    if ctx.overlap_count > 0 {
        let union_m = ctx.unique_algo1 + ctx.unique_algo2 + ctx.overlap_count;
        let uu_t1 = ctx.total_table1.saturating_sub(union_m);
        let uu_t2 = ctx.total_table2.saturating_sub(union_m);
        kv(ws, &mut row, "Matches (Any of A1/A2)", &union_m.to_string())?;
        kv(ws, &mut row, "Unmatched (Any) Table1", &uu_t1.to_string())?;
        kv(ws, &mut row, "Unmatched (Any) Table1 rate", &rate(uu_t1, ctx.total_table1))?;
        kv(ws, &mut row, "Unmatched (Any) Table2", &uu_t2.to_string())?;
        kv(ws, &mut row, "Unmatched (Any) Table2 rate", &rate(uu_t2, ctx.total_table2))?;
    } else {
        kv(ws, &mut row, "Matches (Any of A1/A2)", "N/A (overlap not tracked)")?;
        kv(ws, &mut row, "Unmatched (Any) Table1", "N/A")?;
        kv(ws, &mut row, "Unmatched (Any) Table2", "N/A")?;
    }


    kv(ws, &mut row, "Fetch time (s)", &format!("{:.3}", ctx.fetch_time.as_secs_f64()))?;
    kv(ws, &mut row, "Match A1 time (s)", &format!("{:.3}", ctx.match1_time.as_secs_f64()))?;
    kv(ws, &mut row, "Match A2 time (s)", &format!("{:.3}", ctx.match2_time.as_secs_f64()))?;
    kv(ws, &mut row, "Export time (s)", &format!("{:.3}", ctx.export_time.as_secs_f64()))?;

    kv(ws, &mut row, "Mem used start (MB)", &ctx.mem_used_start_mb.to_string())?;
    kv(ws, &mut row, "Mem used end (MB)", &ctx.mem_used_end_mb.to_string())?;

    kv(ws, &mut row, "Started (UTC)", &ctx.started_utc.to_rfc3339())?;
    kv(ws, &mut row, "Ended (UTC)", &ctx.ended_utc.to_rfc3339())?;
    kv(ws, &mut row, "Duration (s)", &format!("{:.3}", ctx.duration_secs))?;
    kv(ws, &mut row, "Algorithm", &ctx.algo_used)?;
    kv(ws, &mut row, "GPU Used", if ctx.gpu_used { "true" } else { "false" })?;
    kv(ws, &mut row, "GPU Total (MB)", &ctx.gpu_total_mb.to_string())?;
    kv(ws, &mut row, "GPU Free End (MB)", &ctx.gpu_free_mb_end.to_string())?;

    Ok(())
}


pub struct XlsxStreamWriter {
    workbook: Workbook,
    next_r1: u32,
    next_r2: u32,
    out_path: String,
}

impl XlsxStreamWriter {
    pub fn create(out_path: &str) -> Result<Self> {
        let mut workbook = Workbook::new();
        {
            let mut ws1 = workbook.add_worksheet();
            ws1.set_name("Algorithm_1_Results")?;
            write_algo1_sheet(&mut ws1, &[])?;
        }
        {
            let mut ws2 = workbook.add_worksheet();
            ws2.set_name("Algorithm_2_Results")?;
            write_algo2_sheet(&mut ws2, &[])?;
        }
        {
            let ws3 = workbook.add_worksheet();
            ws3.set_name("Summary")?;
        }
        Ok(Self{ workbook, next_r1: 1, next_r2: 1, out_path: out_path.to_string() })
    }
    pub fn append_algo1(&mut self, m: &MatchPair) -> Result<()> {
        let r = self.next_r1; self.next_r1 += 1;
        let even = row_format_even();
        {
            let sheets = self.workbook.worksheets_mut();
            let ws = &mut sheets[0];
            if (r as usize - 1) % 2 == 0 { ws.set_row_format(r, &even)?; }
            ws.write_number(r, 0, m.person1.id as f64)?;
            ws.write_string(r, 1, m.person1.uuid.as_deref().unwrap_or(""))?;
            ws.write_string(r, 2, m.person1.first_name.as_deref().unwrap_or(""))?;
            ws.write_string(r, 3, m.person1.last_name.as_deref().unwrap_or(""))?;
            ws.write_string(r, 4, &m.person1.birthdate.as_ref().map(|d| d.to_string()).unwrap_or_default())?;
            ws.write_number(r, 5, m.person2.id as f64)?;
            ws.write_string(r, 6, m.person2.uuid.as_deref().unwrap_or(""))?;
            ws.write_string(r, 7, m.person2.first_name.as_deref().unwrap_or(""))?;
            ws.write_string(r, 8, m.person2.last_name.as_deref().unwrap_or(""))?;
            ws.write_string(r, 9, &m.person2.birthdate.as_ref().map(|d| d.to_string()).unwrap_or_default())?;
            ws.write_string(r,10, if m.is_matched_infnbd {"true"} else {"false"})?;
            ws.write_number(r,11, m.confidence as f64)?;
            ws.write_string(r,12, &m.matched_fields.join(";"))?;
        }
        Ok(())
    }
    pub fn append_algo2(&mut self, m: &MatchPair) -> Result<()> {
        let r = self.next_r2; self.next_r2 += 1;
        let even = row_format_even();
        {
            let sheets = self.workbook.worksheets_mut();
            let ws = &mut sheets[1];
            if (r as usize - 1) % 2 == 0 { ws.set_row_format(r, &even)?; }
            ws.write_number(r, 0, m.person1.id as f64)?;
            ws.write_string(r, 1, m.person1.uuid.as_deref().unwrap_or(""))?;
            ws.write_string(r, 2, m.person1.first_name.as_deref().unwrap_or(""))?;
            ws.write_string(r, 3, m.person1.middle_name.as_deref().unwrap_or(""))?;
            ws.write_string(r, 4, m.person1.last_name.as_deref().unwrap_or(""))?;
            ws.write_string(r, 5, &m.person1.birthdate.as_ref().map(|d| d.to_string()).unwrap_or_default())?;
            ws.write_number(r, 6, m.person2.id as f64)?;
            ws.write_string(r, 7, m.person2.uuid.as_deref().unwrap_or(""))?;
            ws.write_string(r, 8, m.person2.first_name.as_deref().unwrap_or(""))?;
            ws.write_string(r, 9, m.person2.middle_name.as_deref().unwrap_or(""))?;
            ws.write_string(r,10, m.person2.last_name.as_deref().unwrap_or(""))?;
            ws.write_string(r,11, &m.person2.birthdate.as_ref().map(|d| d.to_string()).unwrap_or_default())?;
            ws.write_string(r,12, if m.is_matched_infnmnbd {"true"} else {"false"})?;
            ws.write_number(r,13, m.confidence as f64)?;
            ws.write_string(r,14, &m.matched_fields.join(";"))?;
        }
        Ok(())
    }
    pub fn finalize(mut self, summary: &SummaryContext) -> Result<()> {
        {
            let sheets = self.workbook.worksheets_mut();
            let ws = &mut sheets[2];
            write_summary_sheet(ws, summary)?;
        }
        self.workbook.save(&self.out_path)?;
        Ok(())
    }
}

pub fn export_to_xlsx(
    algo1_matches: &[MatchPair],
    algo2_matches: &[MatchPair],
    out_path: &str,
    summary: &SummaryContext,
) -> Result<()> {
    ensure_parent_dir(out_path)?;

    let mut workbook = Workbook::new();

    // Algorithm 1 sheet
    let mut sheet1 = workbook.add_worksheet();
    sheet1.set_name("Algorithm_1_Results")?;
    write_algo1_sheet(&mut sheet1, algo1_matches)?;

    // Algorithm 2 sheet
    let mut sheet2 = workbook.add_worksheet();
    sheet2.set_name("Algorithm_2_Results")?;
    write_algo2_sheet(&mut sheet2, algo2_matches)?;

    // Summary sheet
    let mut sheet3 = workbook.add_worksheet();
    sheet3.set_name("Summary")?;
    write_summary_sheet(&mut sheet3, summary)?;

    workbook.save(out_path)?;
    Ok(())
}


pub fn export_summary_xlsx(out_path: &str, summary: &SummaryContext) -> Result<()> {

    ensure_parent_dir(out_path)?;
    let mut workbook = Workbook::new();
    let mut sheet = workbook.add_worksheet();
    sheet.set_name("Summary")?;
    write_summary_sheet(&mut sheet, summary)?;
    workbook.save(out_path)?;
    Ok(())
}

use crate::matching::HouseholdAggRow;

pub fn export_households_xlsx(out_path: &str, rows: &[HouseholdAggRow]) -> Result<()> {
    ensure_parent_dir(out_path)?;
    let mut workbook = Workbook::new();
    let sheet = workbook.add_worksheet();
    sheet.set_name("Household_Matches")?;

    let hfmt = header_format();
    sheet.write_string_with_format(0,0, "id", &hfmt)?;
    sheet.write_string_with_format(0,1, "uuid", &hfmt)?;
    sheet.write_string_with_format(0,2, "hh_id", &hfmt)?;
    sheet.write_string_with_format(0,3, "match_percentage", &hfmt)?;
    sheet.write_string_with_format(0,4, "region_code", &hfmt)?;
    sheet.write_string_with_format(0,5, "poor_hat_0", &hfmt)?;
    sheet.write_string_with_format(0,6, "poor_hat_10", &hfmt)?;

    let even = row_format_even();
    for (i, r) in rows.iter().enumerate() {
        let row = (i as u32) + 1;
        if i % 2 == 0 { sheet.set_row_format(row, &even)?; }
        sheet.write_number(row, 0, r.row_id as f64)?;
        sheet.write_string(row, 1, &r.uuid)?;
        sheet.write_string(row, 2, &r.hh_id)?;
        sheet.write_number(row, 3, r.match_percentage as f64)?;
        sheet.write_string(row, 4, r.region_code.as_deref().unwrap_or(""))?;
        sheet.write_string(row, 5, r.poor_hat_0.as_deref().unwrap_or(""))?;
        sheet.write_string(row, 6, r.poor_hat_10.as_deref().unwrap_or(""))?;
    }

    workbook.save(out_path)?;
    Ok(())
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::Person;
    use chrono::NaiveDate;

    fn p(id: i64, f: &str, m: Option<&str>, l: &str, d: (i32,u32,u32)) -> Person {
        Person{ id, uuid: Some(format!("u{}", id)), first_name: Some(f.into()), middle_name: m.map(|s| s.to_string()), last_name: Some(l.into()), birthdate: NaiveDate::from_ymd_opt(d.0,d.1,d.2), hh_id: None }
    }

    #[test]
    fn write_xlsx_basic() {
        let a1 = vec![MatchPair{
            person1: p(1,"A",None,"Z",(2000,1,1)),
            person2: p(2,"A",None,"Z",(2000,1,1)),
            confidence: 1.0,
            matched_fields: vec!["first_name".into(),"last_name".into(),"birthdate".into()],
            is_matched_infnbd: true,
            is_matched_infnmnbd: false,
        }];
        let a2: Vec<MatchPair> = vec![];
        let out = "./target/test_matches.xlsx";
        let _ = std::fs::remove_file(out);
        let summary = SummaryContext{
            db_name: "db".into(), table1: "t1".into(), table2: "t2".into(),
            total_table1: 1, total_table2: 1, matches_algo1: 1, matches_algo2: 0,
            matches_fuzzy: 0,
            overlap_count: 0, unique_algo1: 1, unique_algo2: 0,
            fetch_time: std::time::Duration::from_millis(1),
            match1_time: std::time::Duration::from_millis(1),
            match2_time: std::time::Duration::from_millis(1),
            export_time: std::time::Duration::from_millis(0),
            mem_used_start_mb: 0, mem_used_end_mb: 0,
            started_utc: chrono::Utc::now(),
            ended_utc: chrono::Utc::now(),
            duration_secs: 0.0,
            algo_used: "Both (1,2)".into(),
            gpu_used: false,
            gpu_total_mb: 0,
            gpu_free_mb_end: 0,
        };
        let res = export_to_xlsx(&a1, &a2, out, &summary);
        assert!(res.is_ok());
        let meta = std::fs::metadata(out).unwrap();
        assert!(meta.len() > 0);
    }
}
