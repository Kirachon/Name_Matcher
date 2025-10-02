#![cfg(feature = "new_engine")]

use chrono::NaiveDate;
use name_matcher::matching::{match_all_progress, MatchingAlgorithm, ProgressConfig};
use name_matcher::engine::person_pipeline::run_new_engine_in_memory;
use name_matcher::models::Person;

fn p(id: i64, first: &str, mid: Option<&str>, last: &str, ymd: (i32,u32,u32)) -> Person {
    Person { id, uuid: Some(format!("u{}", id)), first_name: Some(first.into()), middle_name: mid.map(|s| s.into()), last_name: Some(last.into()), birthdate: Some(NaiveDate::from_ymd_opt(ymd.0, ymd.1, ymd.2).unwrap()) }
}

fn parity_for(algo: MatchingAlgorithm) {
    let a = vec![
        p(1, "John", Some("Q"), "Doe", (2000,1,1)),
        p(2, "Ann", None, "Smith", (1999,12,31)),
        p(3, "Bob", None, "Brown", (1980,5,5)),
    ];
    let b = vec![
        p(10, "John", Some("Q"), "Doe", (2000,1,1)),
        p(20, "Ann", None, "Smith", (1999,12,31)),
        p(30, "Alice", None, "Brown", (1980,5,5)),
    ];
    let legacy = match_all_progress(&a, &b, algo, ProgressConfig::default(), |_u| {});
    let new_eng = run_new_engine_in_memory(&a, &b, algo);
    let set_legacy: std::collections::BTreeSet<(i64,i64)> = legacy.iter().map(|m| (m.person1.id, m.person2.id)).collect();
    let set_new:    std::collections::BTreeSet<(i64,i64)> = new_eng.iter().map(|m| (m.person1.id, m.person2.id)).collect();
    assert_eq!(set_legacy, set_new, "parity mismatch for {:?}", algo);
}

#[test]
fn parity_algo1() { parity_for(MatchingAlgorithm::IdUuidYasIsMatchedInfnbd); }
#[test]
fn parity_algo2() { parity_for(MatchingAlgorithm::IdUuidYasIsMatchedInfnmnbd); }
#[test]
fn parity_fuzzy() { parity_for(MatchingAlgorithm::Fuzzy); }
#[test]
fn parity_fuzzy_no_mid() { parity_for(MatchingAlgorithm::FuzzyNoMiddle); }

