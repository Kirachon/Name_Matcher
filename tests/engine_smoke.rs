#![cfg(feature = "new_engine")]

use name_matcher::engine::{Matcher, Partitioner, Checkpointer, StreamEngine};

#[derive(Debug)]
struct DummyMatcher;
impl Matcher for DummyMatcher {
    type ItemA = u32;
    type ItemB = u32;
    type MatchExplanation = ();
    fn compare(&self, a: &u32, b: &u32) -> Option<(u32, Self::MatchExplanation)> {
        if a == b { Some((100, ())) } else { None }
    }
}

#[derive(Debug)]
struct ModPartitioner(u32);
impl Partitioner<u32> for ModPartitioner {
    fn key(&self, item: &u32) -> String { format!("{}", item % self.0) }
}

#[derive(Default)]
struct MemoryCk(std::collections::HashMap<(String, String), String>);
impl Checkpointer for MemoryCk {
    fn save(&mut self, job: &str, partition: &str, token: &str) -> anyhow::Result<()> {
        self.0.insert((job.to_string(), partition.to_string()), token.to_string());
        Ok(())
    }
    fn load(&self, job: &str, partition: &str) -> anyhow::Result<Option<String>> {
        Ok(self.0.get(&(job.to_string(), partition.to_string())).cloned())
    }
}

#[test]
fn engine_scaffolding_compiles_and_runs() {
    let matcher = DummyMatcher;
    let pa = ModPartitioner(4);
    let pb = ModPartitioner(4);
    let mut ck = MemoryCk::default();
    let mut engine = StreamEngine::new(matcher, pa, pb, ck);
    let a = vec![1u32, 2, 3, 4];
    let b = vec![2u32, 3, 4, 5];
    engine.stream_match("job1", a.iter(), b.iter()).unwrap();
}

