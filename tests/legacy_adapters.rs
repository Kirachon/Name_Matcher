#![cfg(feature = "new_engine")]

use chrono::NaiveDate;
use name_matcher::engine::legacy_adapters::{
    LegacyAdapterAlgo1, LegacyAdapterAlgo2, LegacyAdapterFuzzy, LegacyAdapterFuzzyNoMiddle
};
use name_matcher::engine::Matcher as EngineMatcher;
use name_matcher::models::Person;

fn p(id: i64, first: &str, mid: Option<&str>, last: &str, ymd: (i32,u32,u32)) -> Person {
    Person { id, uuid: Some(format!("u{}", id)), first_name: Some(first.into()), middle_name: mid.map(|s| s.into()), last_name: Some(last.into()), birthdate: Some(NaiveDate::from_ymd_opt(ymd.0, ymd.1, ymd.2).unwrap()) }
}

#[test]
fn adapter_algo1_matches_equal_names_and_birthdate() {
    let a = p(1, "John", None, "Doe", (2000,1,1));
    let b = p(2, "John", None, "Doe", (2000,1,1));
    let m = LegacyAdapterAlgo1::default();
    assert!(m.compare(&a, &b).is_some());
}

#[test]
fn adapter_algo2_requires_matching_middle_or_both_missing() {
    let a = p(1, "John", Some("Q"), "Doe", (2000,1,1));
    let b = p(2, "John", Some("Q"), "Doe", (2000,1,1));
    let m = LegacyAdapterAlgo2::default();
    assert!(m.compare(&a, &b).is_some());
}

#[test]
fn adapter_fuzzy_direct_match_equal_fullname_and_birthdate() {
    let a = p(1, "Ann", None, "Smith", (1999,12,31));
    let b = p(2, "Ann", None, "Smith", (1999,12,31));
    let m = LegacyAdapterFuzzy::default();
    let out = m.compare(&a, &b);
    assert!(out.is_some());
    let (score, label) = out.unwrap();
    assert_eq!(score, 100);
    assert!(label.to_ascii_uppercase().contains("MATCH") || label.to_ascii_uppercase().contains("CASE"));
}

#[test]
fn adapter_fuzzy_no_mid_direct_match_equal_fullname_and_birthdate() {
    let a = p(1, "Ann", None, "Smith", (1999,12,31));
    let b = p(2, "Ann", None, "Smith", (1999,12,31));
    let m = LegacyAdapterFuzzyNoMiddle::default();
    let out = m.compare(&a, &b);
    assert!(out.is_some());
    let (score, _label) = out.unwrap();
    assert_eq!(score, 100);
}

#[test]
fn fuzzy_birthdate_mismatch_should_not_match() {
    let a = p(1, "Ann", None, "Smith", (1999,12,31));
    let b = p(2, "Ann", None, "Smith", (1998,1,1));
    let m = LegacyAdapterFuzzyNoMiddle::default();
    assert!(m.compare(&a, &b).is_none());
}

