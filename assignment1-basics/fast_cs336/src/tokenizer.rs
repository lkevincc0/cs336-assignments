use std::collections::HashMap;
use std::sync::Arc;
use std::io::{BufRead, BufReader};
use std::fs::File;

use dashmap::DashMap;
use fancy_regex::Regex;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use once_cell::sync::Lazy;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use rayon::prelude::*;

/// GPT-2 tokenization pattern
static GPT2_PAT: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")
        .expect("invalid GPT-2 regex")
});

#[pyclass]
pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, usize>,
    _merges: Vec<(Vec<u8>, Vec<u8>)>,
    merge_ranks: HashMap<(Vec<u8>, Vec<u8>), usize>,
    special_re: Option<Regex>,
    merge_cache: Arc<DashMap<String, Vec<Vec<u8>>>>,
}

unsafe impl Send for Tokenizer {}
unsafe impl Sync for Tokenizer {}

impl Tokenizer {
    fn build_merge_ranks(merges: &[(Vec<u8>, Vec<u8>)]) -> HashMap<(Vec<u8>, Vec<u8>), usize> {
        merges.iter().enumerate().map(|(i, m)| ((m.0.clone(), m.1.clone()), i)).collect()
    }

    fn build_special_re(special_tokens: &[String]) -> Option<Regex> {
        if special_tokens.is_empty() { return None; }
        let mut sorted = special_tokens.to_vec();
        sorted.sort_by(|a, b| b.len().cmp(&a.len()));
        let pattern = sorted.iter()
            .map(|s| fancy_regex::escape(s))
            .collect::<Vec<_>>()
            .join("|");
        Regex::new(&pattern).ok()
    }

    fn merge_word(&self, mut word_bytes: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        while word_bytes.len() > 1 {
            let mut best: Option<((Vec<u8>, Vec<u8>), usize)> = None;
            for i in 0..(word_bytes.len() - 1) {
                let pair = (word_bytes[i].clone(), word_bytes[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    match &best {
                        Some((_, br)) if *br <= rank => {}
                        _ => best = Some((pair, rank)),
                    }
                }
            }
            let best_pair = match best { None => break, Some((p, _)) => p };
            let mut new_word: Vec<Vec<u8>> = Vec::with_capacity(word_bytes.len());
            let mut i = 0;
            while i < word_bytes.len() {
                if i + 1 < word_bytes.len() && word_bytes[i] == best_pair.0 && word_bytes[i + 1] == best_pair.1 {
                    let mut merged = word_bytes[i].clone();
                    merged.extend_from_slice(&word_bytes[i + 1]);
                    new_word.push(merged);
                    i += 2;
                } else {
                    new_word.push(word_bytes[i].clone());
                    i += 1;
                }
            }
            word_bytes = new_word;
        }
        word_bytes
    }

    /// Pure-Rust encode — no Python callbacks, safe to call without the GIL.
    pub fn encode_text(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        let parts: Vec<(&str, bool)> = match &self.special_re {
            None => vec![(text, false)],
            Some(re) => {
                let mut result = Vec::new();
                let mut last_end = 0usize;
                for m in re.find_iter(text) {
                    let Ok(m) = m else { continue };
                    if m.start() > last_end { result.push((&text[last_end..m.start()], false)); }
                    result.push((m.as_str(), true));
                    last_end = m.end();
                }
                if last_end < text.len() { result.push((&text[last_end..], false)); }
                result
            }
        };
        for (part, is_special) in parts {
            if part.is_empty() { continue; }
            if is_special {
                if let Some(&id) = self.token_to_id.get(part.as_bytes()) { ids.push(id); }
                continue;
            }
            for m in GPT2_PAT.find_iter(part) {
                let Ok(m) = m else { continue };
                let word_str = m.as_str();
                let merged = if let Some(cached) = self.merge_cache.get(word_str) {
                    cached.clone()
                } else {
                    let word_bytes: Vec<Vec<u8>> = word_str.bytes().map(|b| vec![b]).collect();
                    let merged = self.merge_word(word_bytes);
                    self.merge_cache.insert(word_str.to_string(), merged.clone());
                    merged
                };
                for b in merged.iter() {
                    if let Some(&id) = self.token_to_id.get(b) { ids.push(id); }
                }
            }
        }
        ids
    }

    fn from_vocab_merges(vocab_map: HashMap<i64, Vec<u8>>, merges_vec: Vec<(Vec<u8>, Vec<u8>)>, special_tokens_vec: Vec<String>) -> Self {
        let max_id = vocab_map.keys().copied().max().unwrap_or(-1);
        let size = (max_id + 1) as usize;
        let mut vocab_vec: Vec<Vec<u8>> = vec![Vec::new(); size];
        let mut token_to_id: HashMap<Vec<u8>, usize> = HashMap::new();
        for (k, v) in vocab_map {
            if k < 0 { continue; }
            let idx = k as usize;
            if idx >= vocab_vec.len() { vocab_vec.resize(idx + 1, Vec::new()); }
            token_to_id.insert(v.clone(), idx);
            vocab_vec[idx] = v;
        }
        let merge_ranks = Self::build_merge_ranks(&merges_vec);
        let special_re = Self::build_special_re(&special_tokens_vec);
        Tokenizer { vocab: vocab_vec, token_to_id, _merges: merges_vec, merge_ranks, special_re, merge_cache: Arc::new(DashMap::new()) }
    }
}

#[pymethods]
impl Tokenizer {
    #[new]
    fn new(vocab: Option<HashMap<i64, Vec<u8>>>, merges: Option<Vec<(Vec<u8>, Vec<u8>)>>, special_tokens: Option<Vec<String>>) -> Self {
        Self::from_vocab_merges(vocab.unwrap_or_default(), merges.unwrap_or_default(), special_tokens.unwrap_or_default())
    }

    #[staticmethod]
    fn from_files(py: Python, vocab_filepath: &str, merges_filepath: &str, special_tokens: Option<Vec<String>>) -> PyResult<Tokenizer> {
        let pickle = py.import("pickle").map_err(|e| PyValueError::new_err(e.to_string()))?;
        let builtins = py.import("builtins").map_err(|e| PyValueError::new_err(e.to_string()))?;
        let f_vocab = builtins.call_method1("open", (vocab_filepath, "rb")).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let vocab_py = pickle.call_method1("load", (f_vocab,)).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let f_merges = builtins.call_method1("open", (merges_filepath, "rb")).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let merges_py = pickle.call_method1("load", (f_merges,)).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let vocab_map: HashMap<i64, Vec<u8>> = vocab_py.extract::<HashMap<i64, Vec<u8>>>().map_err(|e| PyValueError::new_err(e.to_string()))?;
        let merges_vec: Vec<(Vec<u8>, Vec<u8>)> = merges_py.extract::<Vec<(Vec<u8>, Vec<u8>)>>().map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self::from_vocab_merges(vocab_map, merges_vec, special_tokens.unwrap_or_default()))
    }

    fn decode(&self, ids: Vec<usize>) -> PyResult<String> {
        let mut out: Vec<u8> = Vec::new();
        for id in ids {
            if id >= self.vocab.len() { return Err(PyValueError::new_err(format!("id {} out of vocab range", id))); }
            out.extend_from_slice(&self.vocab[id]);
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    /// Encode a single string. Pure Rust — no Python callbacks.
    fn encode(&self, text: String) -> Vec<usize> {
        self.encode_text(&text)
    }

    // Encode a Python iterable in parallel via Rayon.
    fn encode_iterable(&self, py: Python, iterable: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
        let lines: Vec<String> = iterable
            .try_iter()?
            .map(|item| item?.extract::<String>())
            .collect::<PyResult<Vec<_>>>()?;
        let _ = py; // no more Python calls needed
        let all_ids: Vec<Vec<usize>> = lines.par_iter().map(|line| self.encode_text(line)).collect();
        Ok(all_ids.into_iter().flatten().collect())
    }

    // Encode a file directly in Rust
    fn encode_file(&self, path: &str) -> PyResult<Vec<usize>> {
        let file = File::open(path)
            .map_err(|e| PyValueError::new_err(format!("cannot open file: {}", e)))?;
        let total_bytes = file.metadata().map(|m| m.len()).unwrap_or(0);

        let mp = MultiProgress::new();
        let byte_style = ProgressStyle::with_template(
            "[1/2] Reading  {bar:40.cyan/blue} {bytes}/{total_bytes} [{elapsed_precise} < {eta}] {bytes_per_sec}",
        ).unwrap().progress_chars("█▉▊▋▌▍▎▏ ");
        let pb_read = mp.add(ProgressBar::new(total_bytes));
        pb_read.set_style(byte_style);

        let reader = BufReader::new(file);
        let mut lines: Vec<String> = Vec::new();
        for line in reader.lines() {
            let line = line.map_err(|e| PyValueError::new_err(format!("read error: {}", e)))?;
            pb_read.inc(line.len() as u64 + 1); // +1 for newline
            lines.push(line);
        }
        pb_read.finish_with_message("done");

        let line_style = ProgressStyle::with_template(
            "[2/2] Encoding {bar:40.green/blue} {pos}/{len} lines [{elapsed_precise} < {eta}] {per_sec}",
        ).unwrap().progress_chars("█▉▊▋▌▍▎▏ ");
        let pb_enc = mp.add(ProgressBar::new(lines.len() as u64));
        pb_enc.set_style(line_style);

        let all_ids: Vec<Vec<usize>> = lines.par_iter().map(|line| {
            let result = self.encode_text(line);
            pb_enc.inc(1);
            result
        }).collect();
        pb_enc.finish_with_message("done");

        Ok(all_ids.into_iter().flatten().collect())
    }
}
