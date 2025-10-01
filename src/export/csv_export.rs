use crate::matching::{MatchPair, MatchingAlgorithm};
use crate::export::xlsx_export::SummaryContext;
use anyhow::Result;
use csv::Writer;

pub fn export_to_csv(results: &[MatchPair], path: &str, algorithm: MatchingAlgorithm, fuzzy_min_confidence: f32) -> Result<()> {
    use std::fs::File;
    use std::io::BufWriter;
    let file = File::create(path)?;
    let buf = BufWriter::with_capacity(64 * 1024, file);
    let mut w = Writer::from_writer(buf);
    write_headers(&mut w, algorithm)?;
    for pair in results { write_pair(&mut w, pair, algorithm, fuzzy_min_confidence)?; }
    w.flush()?;
    Ok(())
}

fn write_headers<W: std::io::Write>(w: &mut Writer<W>, algorithm: MatchingAlgorithm) -> Result<()> {
    let headers = match algorithm {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => vec![
            "Table1_ID", "Table1_UUID", "Table1_FirstName", "Table1_LastName", "Table1_Birthdate",
            "Table2_ID", "Table2_UUID", "Table2_FirstName", "Table2_LastName", "Table2_Birthdate",
            "is_matched_Infnbd", "Confidence", "MatchedFields",
        ],
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => vec![
            "Table1_ID", "Table1_UUID", "Table1_FirstName", "Table1_MiddleName", "Table1_LastName", "Table1_Birthdate",
            "Table2_ID", "Table2_UUID", "Table2_FirstName", "Table2_MiddleName", "Table2_LastName", "Table2_Birthdate",
            "is_matched_Infnmnbd", "Confidence", "MatchedFields",
        ],
        MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle | MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6 => vec![
            "Table1_ID", "Table1_UUID", "Table1_FirstName", "Table1_MiddleName", "Table1_LastName", "Table1_Birthdate",
            "Table2_ID", "Table2_UUID", "Table2_FirstName", "Table2_MiddleName", "Table2_LastName", "Table2_Birthdate",
            "is_matched_Fuzzy", "Confidence", "MatchedFields",
        ],
        MatchingAlgorithm::FuzzyBirthdate => { anyhow::bail!("Algorithm 7 (FuzzyBirthdate) has been deprecated and is not supported for export") }
    };
    w.write_record(&headers)?;
    Ok(())
}

fn write_pair<W: std::io::Write>(w: &mut Writer<W>, pair: &MatchPair, algorithm: MatchingAlgorithm, fuzzy_min_confidence: f32) -> Result<()> {
    match algorithm {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
            w.write_record(&[
                pair.person1.id.to_string(),
                pair.person1.uuid.clone().unwrap_or_default(),
                pair.person1.first_name.clone().unwrap_or_default(),
                pair.person1.last_name.clone().unwrap_or_default(),
                pair.person1.birthdate.map(|d| d.to_string()).unwrap_or_default(),
                pair.person2.id.to_string(),
                pair.person2.uuid.clone().unwrap_or_default(),
                pair.person2.first_name.clone().unwrap_or_default(),
                pair.person2.last_name.clone().unwrap_or_default(),
                pair.person2.birthdate.map(|d| d.to_string()).unwrap_or_default(),
                pair.is_matched_infnbd.to_string(),
                pair.confidence.to_string(),
                pair.matched_fields.join(","),
            ])?;
        }
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
            w.write_record(&[
                pair.person1.id.to_string(),
                pair.person1.uuid.clone().unwrap_or_default(),
                pair.person1.first_name.clone().unwrap_or_default(),
                pair.person1.middle_name.clone().unwrap_or_default(),
                pair.person1.last_name.clone().unwrap_or_default(),
                pair.person1.birthdate.map(|d| d.to_string()).unwrap_or_default(),
                pair.person2.id.to_string(),
                pair.person2.uuid.clone().unwrap_or_default(),
                pair.person2.first_name.clone().unwrap_or_default(),
                pair.person2.middle_name.clone().unwrap_or_default(),
                pair.person2.last_name.clone().unwrap_or_default(),
                pair.person2.birthdate.map(|d| d.to_string()).unwrap_or_default(),
                pair.is_matched_infnmnbd.to_string(),
                pair.confidence.to_string(),
                pair.matched_fields.join(","),
            ])?;
        }
        MatchingAlgorithm::Fuzzy | MatchingAlgorithm::FuzzyNoMiddle => {
            if pair.confidence < fuzzy_min_confidence {
                // Skip writing fuzzy matches below selected threshold
                return Ok(());
            }
            w.write_record(&[
                pair.person1.id.to_string(),
                pair.person1.uuid.clone().unwrap_or_default(),
                pair.person1.first_name.clone().unwrap_or_default(),
                pair.person1.middle_name.clone().unwrap_or_default(),
                pair.person1.last_name.clone().unwrap_or_default(),
                pair.person1.birthdate.map(|d| d.to_string()).unwrap_or_default(),
                pair.person2.id.to_string(),
                pair.person2.uuid.clone().unwrap_or_default(),
                pair.person2.first_name.clone().unwrap_or_default(),
                pair.person2.middle_name.clone().unwrap_or_default(),
                pair.person2.last_name.clone().unwrap_or_default(),
                pair.person2.birthdate.map(|d| d.to_string()).unwrap_or_default(),
                "true".into(),
                pair.confidence.to_string(),
                pair.matched_fields.join(","),
            ])?;
        }
        MatchingAlgorithm::HouseholdGpu | MatchingAlgorithm::HouseholdGpuOpt6 => {
            // Not applicable for person-level writer
            return Ok(());
        }
        MatchingAlgorithm::FuzzyBirthdate => { anyhow::bail!("Algorithm 7 (FuzzyBirthdate) has been deprecated and is not supported for export") }
    }
    Ok(())
}


use crate::matching::HouseholdAggRow;

pub struct HouseholdCsvWriter {
    writer: Writer<std::fs::File>,
}

impl HouseholdCsvWriter {
    pub fn create(path: &str) -> Result<Self> {
        let mut w = Writer::from_path(path)?;
        w.write_record(&["id","uuid","hh_id","match_percentage","region_code","poor_hat_0","poor_hat_10"])?;
        Ok(Self{ writer: w })
    }
    pub fn write(&mut self, row: &HouseholdAggRow) -> Result<()> {
        self.writer.write_record(&[
            row.row_id.to_string(),
            row.uuid.clone(),
            row.hh_id.to_string(),
            format!("{:.2}", row.match_percentage),
            row.region_code.clone().unwrap_or_default(),
            row.poor_hat_0.clone().unwrap_or_default(),
            row.poor_hat_10.clone().unwrap_or_default(),
        ])?; Ok(())
    }
    pub fn flush(mut self) -> Result<()> { self.writer.flush()?; Ok(()) }
}

pub struct CsvStreamWriter {
    writer: Writer<std::fs::File>,
    algo: MatchingAlgorithm,
    fuzzy_min_confidence: f32,
}

impl CsvStreamWriter {
    pub fn create(path: &str, algorithm: MatchingAlgorithm, fuzzy_min_confidence: f32) -> Result<Self> {
        let mut writer = Writer::from_path(path)?;
        write_headers(&mut writer, algorithm)?;
        Ok(Self { writer, algo: algorithm, fuzzy_min_confidence })
    }
    pub fn write(&mut self, pair: &MatchPair) -> Result<()> { write_pair(&mut self.writer, pair, self.algo, self.fuzzy_min_confidence) }
    pub fn flush_partial(&mut self) -> Result<()> { self.writer.flush()?; Ok(()) }
    pub fn flush(mut self) -> Result<()> { self.writer.flush()?; Ok(()) }
}

pub fn export_summary_csv(path: &str, ctx: &SummaryContext) -> Result<()> {
    let mut w = Writer::from_path(path)?;
    w.write_record(["Key","Value"])?;

    let mut write_kv = |k: &str, v: String| -> Result<()> {
        w.write_record(&[k, v.as_str()])?; Ok(())
    };

    write_kv("Database", ctx.db_name.clone())?;
    write_kv("Table 1", ctx.table1.clone())?;
    write_kv("Table 2", ctx.table2.clone())?;
    write_kv("Total records (Table1)", ctx.total_table1.to_string())?;
    write_kv("Total records (Table2)", ctx.total_table2.to_string())?;
    write_kv("Matches (Algorithm 1)", ctx.matches_algo1.to_string())?;
    write_kv("Matches (Algorithm 2)", ctx.matches_algo2.to_string())?;
    write_kv("Matches (Fuzzy)", ctx.matches_fuzzy.to_string())?;
    write_kv("Overlap (A1 ∩ A2)", ctx.overlap_count.to_string())?;
    write_kv("Unique to A1", ctx.unique_algo1.to_string())?;
    write_kv("Unique to A2", ctx.unique_algo2.to_string())?;
    // Derived stats
    let u1_t1 = ctx.total_table1.saturating_sub(ctx.matches_algo1);
    let u1_t2 = ctx.total_table2.saturating_sub(ctx.matches_algo1);
    let u2_t1 = ctx.total_table1.saturating_sub(ctx.matches_algo2);
    let u2_t2 = ctx.total_table2.saturating_sub(ctx.matches_algo2);
    let rate = |num: usize, den: usize| if den == 0 { "0.000".to_string() } else { format!("{:.3}", (num as f64)/(den as f64)) };
    write_kv("Unmatched by A1 (Table1)", u1_t1.to_string())?;
    write_kv("Unmatched by A1 (Table1) rate", rate(u1_t1, ctx.total_table1))?;
    write_kv("Unmatched by A1 (Table2)", u1_t2.to_string())?;
    write_kv("Unmatched by A1 (Table2) rate", rate(u1_t2, ctx.total_table2))?;
    write_kv("Unmatched by A2 (Table1)", u2_t1.to_string())?;
    write_kv("Unmatched by A2 (Table1) rate", rate(u2_t1, ctx.total_table1))?;
    write_kv("Unmatched by A2 (Table2)", u2_t2.to_string())?;
    write_kv("Unmatched by A2 (Table2) rate", rate(u2_t2, ctx.total_table2))?;

    // Fuzzy unmatched when available
    let uf_t1 = ctx.total_table1.saturating_sub(ctx.matches_fuzzy);
    let uf_t2 = ctx.total_table2.saturating_sub(ctx.matches_fuzzy);
    write_kv("Unmatched by Fuzzy (Table1)", uf_t1.to_string())?;
    write_kv("Unmatched by Fuzzy (Table2)", uf_t2.to_string())?;

    // Union metrics if overlap recorded
    if ctx.overlap_count > 0 {
        let union_m = ctx.unique_algo1 + ctx.unique_algo2 + ctx.overlap_count;
        let uu_t1 = ctx.total_table1.saturating_sub(union_m);
        let uu_t2 = ctx.total_table2.saturating_sub(union_m);
        write_kv("Matches (Any of A1/A2)", union_m.to_string())?;
        write_kv("Unmatched (Any) Table1", uu_t1.to_string())?;
        write_kv("Unmatched (Any) Table1 rate", rate(uu_t1, ctx.total_table1))?;
        write_kv("Unmatched (Any) Table2", uu_t2.to_string())?;
        write_kv("Unmatched (Any) Table2 rate", rate(uu_t2, ctx.total_table2))?;
    } else {
        write_kv("Matches (Any of A1/A2)", "N/A (overlap not tracked)".to_string())?;
        write_kv("Unmatched (Any) Table1", "N/A".to_string())?;
        write_kv("Unmatched (Any) Table2", "N/A".to_string())?;
    }

    write_kv("Fetch time (s)", format!("{:.3}", ctx.fetch_time.as_secs_f64()))?;
    write_kv("Match A1 time (s)", format!("{:.3}", ctx.match1_time.as_secs_f64()))?;
    write_kv("Match A2 time (s)", format!("{:.3}", ctx.match2_time.as_secs_f64()))?;
    write_kv("Export time (s)", format!("{:.3}", ctx.export_time.as_secs_f64()))?;
    write_kv("Mem used start (MB)", ctx.mem_used_start_mb.to_string())?;
    write_kv("Mem used end (MB)", ctx.mem_used_end_mb.to_string())?;
    write_kv("Started (UTC)", ctx.started_utc.to_rfc3339())?;
    write_kv("Ended (UTC)", ctx.ended_utc.to_rfc3339())?;
    write_kv("Duration (s)", format!("{:.3}", ctx.duration_secs))?;
    write_kv("Algorithm", ctx.algo_used.clone())?;
    write_kv("GPU Used", if ctx.gpu_used { "true".to_string() } else { "false".to_string() })?;
    write_kv("GPU Total (MB)", ctx.gpu_total_mb.to_string())?;
    write_kv("GPU Free End (MB)", ctx.gpu_free_mb_end.to_string())?;

    w.flush()?;
    Ok(())
}

