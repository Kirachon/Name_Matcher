use crate::matching::{MatchPair, MatchingAlgorithm};
use anyhow::Result;
use csv::Writer;

pub fn export_to_csv(results: &[MatchPair], path: &str, algorithm: MatchingAlgorithm) -> Result<()> {
    let mut w = Writer::from_path(path)?;
    write_headers(&mut w, algorithm)?;
    for pair in results { write_pair(&mut w, pair, algorithm)?; }
    w.flush()?;
    Ok(())
}

fn write_headers(w: &mut Writer<std::fs::File>, algorithm: MatchingAlgorithm) -> Result<()> {
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
        MatchingAlgorithm::Fuzzy => vec![
            "Table1_ID", "Table1_UUID", "Table1_FirstName", "Table1_MiddleName", "Table1_LastName", "Table1_Birthdate",
            "Table2_ID", "Table2_UUID", "Table2_FirstName", "Table2_MiddleName", "Table2_LastName", "Table2_Birthdate",
            "is_matched_Fuzzy", "Confidence", "MatchedFields",
        ],
    };
    w.write_record(&headers)?;
    Ok(())
}

fn write_pair(w: &mut Writer<std::fs::File>, pair: &MatchPair, algorithm: MatchingAlgorithm) -> Result<()> {
    match algorithm {
        MatchingAlgorithm::IdUuidYasIsMatchedInfnbd => {
            w.write_record(&[
                pair.person1.id.to_string(),
                pair.person1.uuid.clone(),
                pair.person1.first_name.clone(),
                pair.person1.last_name.clone(),
                pair.person1.birthdate.to_string(),
                pair.person2.id.to_string(),
                pair.person2.uuid.clone(),
                pair.person2.first_name.clone(),
                pair.person2.last_name.clone(),
                pair.person2.birthdate.to_string(),
                pair.is_matched_infnbd.to_string(),
                pair.confidence.to_string(),
                pair.matched_fields.join(","),
            ])?;
        }
        MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd => {
            w.write_record(&[
                pair.person1.id.to_string(),
                pair.person1.uuid.clone(),
                pair.person1.first_name.clone(),
                pair.person1.middle_name.clone().unwrap_or_default(),
                pair.person1.last_name.clone(),
                pair.person1.birthdate.to_string(),
                pair.person2.id.to_string(),
                pair.person2.uuid.clone(),
                pair.person2.first_name.clone(),
                pair.person2.middle_name.clone().unwrap_or_default(),
                pair.person2.last_name.clone(),
                pair.person2.birthdate.to_string(),
                pair.is_matched_infnmnbd.to_string(),
                pair.confidence.to_string(),
                pair.matched_fields.join(","),
            ])?;
        }
        MatchingAlgorithm::Fuzzy => {
            if pair.confidence < 0.95 {
                // Skip writing fuzzy matches below 95% per new requirement
                return Ok(());
            }
            w.write_record(&[
                pair.person1.id.to_string(),
                pair.person1.uuid.clone(),
                pair.person1.first_name.clone(),
                pair.person1.middle_name.clone().unwrap_or_default(),
                pair.person1.last_name.clone(),
                pair.person1.birthdate.to_string(),
                pair.person2.id.to_string(),
                pair.person2.uuid.clone(),
                pair.person2.first_name.clone(),
                pair.person2.middle_name.clone().unwrap_or_default(),
                pair.person2.last_name.clone(),
                pair.person2.birthdate.to_string(),
                "true".into(),
                pair.confidence.to_string(),
                pair.matched_fields.join(","),
            ])?;
        }
    }
    Ok(())
}

pub struct CsvStreamWriter {
    writer: Writer<std::fs::File>,
    algo: MatchingAlgorithm,
}

impl CsvStreamWriter {
    pub fn create(path: &str, algorithm: MatchingAlgorithm) -> Result<Self> {
        let mut writer = Writer::from_path(path)?;
        write_headers(&mut writer, algorithm)?;
        Ok(Self { writer, algo: algorithm })
    }
    pub fn write(&mut self, pair: &MatchPair) -> Result<()> { write_pair(&mut self.writer, pair, self.algo) }
    pub fn flush_partial(&mut self) -> Result<()> { self.writer.flush()?; Ok(()) }
    pub fn flush(mut self) -> Result<()> { self.writer.flush()?; Ok(()) }
}
