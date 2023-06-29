use indicatif::ParallelProgressIterator;
use indicatif::ProgressIterator;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use rayon::prelude::*;
use serde;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

extern crate num_traits;

use num_traits::float::Float;

#[allow(dead_code)]
#[derive(Deserialize, Debug)]
struct Choice {
    text: String,
    label: bool,
    label_str: String,
    ngrams: Vec<String>,
    #[serde(flatten)]
    extra: HashMap<String, Value>,
}

#[derive(Deserialize, Debug)]
struct DatasetItem {
    choices: Vec<Choice>,
}

// Custom error wrapper
#[derive(Debug)]
struct SerdeJsonError(serde_json::Error);

// Implement `From<SerdeJsonError>` trait for `PyErr`
impl From<SerdeJsonError> for PyErr {
    fn from(err: SerdeJsonError) -> PyErr {
        PyValueError::new_err(err.0.to_string())
    }
}

#[pyfunction]
fn count_ngram_cooccurrences(raw_dataset: &str) -> PyResult<HashMap<(String, String), usize>> {
    let dataset: Vec<DatasetItem> = serde_json::from_str(raw_dataset).map_err(SerdeJsonError)?;

    let mut ngram_label_count: HashMap<(String, String), usize> = HashMap::new();

    for item in dataset {
        for choice in item.choices {
            let label = choice.label_str;
            for ngram in choice.ngrams {
                let ngram_label = (ngram, label.clone());

                // Update count for ngram-label pair
                *ngram_label_count.entry(ngram_label).or_insert(0) += 1;
            }
        }
    }

    Ok(ngram_label_count)
}

#[pyfunction]
fn fast_count_ngram_cooccurrences(raw_dataset: &str) -> PyResult<HashMap<(String, String), usize>> {
    let dataset: Vec<DatasetItem> = serde_json::from_str(raw_dataset).map_err(SerdeJsonError)?;

    let ngram_label_count = dataset
        .par_iter()
        .progress_count(dataset.len() as u64)
        .fold(
            || HashMap::new(),
            |mut acc, item| {
                for choice in &item.choices {
                    let label = &choice.label_str;
                    for ngram in &choice.ngrams {
                        let ngram_label = (ngram.clone(), label.clone());

                        // Update count for ngram-label pair
                        *acc.entry(ngram_label).or_insert(0) += 1;
                    }
                }
                acc
            },
        )
        .reduce(
            || HashMap::new(),
            |a, b| {
                a.into_iter()
                    .chain(b)
                    .fold(HashMap::new(), |mut acc, ((ngram, label), count)| {
                        *acc.entry((ngram, label)).or_insert(0) += count;
                        acc
                    })
            },
        );

    Ok(ngram_label_count)
}

#[pyfunction]
fn get_ngrams(tokens: Vec<&str>, n: usize) -> Vec<String> {
    if n == 0 || n > tokens.len() {
        return vec![];
    }

    tokens
        .par_windows(n)
        .map(|window| window.join(" "))
        .collect()
}

#[pyfunction]
fn get_all_ngrams(tokens: Vec<&str>, n: usize) -> Vec<String> {
    let mut all_ngrams = vec![];

    for i in 1..=n {
        let ngrams = get_ngrams(tokens.clone(), i);
        all_ngrams.extend(ngrams);
    }

    all_ngrams
}

#[pyfunction]
fn get_n_from_ngrams(ngram: &str) -> (String, usize) {
    let tokens: Vec<&str> = ngram.split_whitespace().collect();
    let words_count = tokens.len();
    (ngram.to_string(), words_count)
}

#[pyfunction]
fn get_n_from_ngram_list(ngrams: Vec<String>) -> Vec<(String, usize)> {
    ngrams
        .into_iter()
        .map(|ngram| {
            let n = ngram.split_whitespace().count();
            (ngram, n)
        })
        .collect()
}

#[pyfunction]
fn count_ngram_occurrences(
    ngram_counts: HashMap<(String, String), usize>,
    ngram_to_find: &str,
) -> usize {
    ngram_counts
        .par_iter()
        .filter_map(|((ngram, _label), count)| {
            if ngram == ngram_to_find {
                Some(count)
            } else {
                None
            }
        })
        .sum()
}

#[pyfunction]
fn count_label_occurrences(
    ngram_counts: HashMap<(String, String), usize>,
    label_to_find: &str,
) -> usize {
    ngram_counts
        .par_iter()
        .filter_map(|((_ngram, label), count)| {
            if label == label_to_find {
                Some(count)
            } else {
                None
            }
        })
        .sum()
}

#[pyfunction]
fn get_vocab_size_total_counts(
    ngram_cooccurrences: HashMap<(String, String), usize>,
) -> HashMap<String, usize> {
    let total_cooccurrence_count: usize = ngram_cooccurrences.values().sum();
    let unique_ngrams: HashSet<String> = ngram_cooccurrences
        .keys()
        .progress_count(ngram_cooccurrences.len() as u64)
        .map(|(ngram, _label)| ngram.to_owned())
        .collect();

    let mut results: HashMap<String, usize> = HashMap::new();

    results.insert(String::from("vocab_size"), unique_ngrams.len());
    results.insert(String::from("total_size"), total_cooccurrence_count);
    results
}

#[pyfunction]
fn compute_pmi(
    ngram_cooccurrences: HashMap<(String, String), usize>,
    ngram: &str,
    label: &str,
    vocab_size: usize,
    total_count: usize,
    smoothing: f64,
) -> f64 {
    let ngram_label_count = *ngram_cooccurrences
        .get(&(ngram.to_string(), label.to_string()))
        .unwrap_or(&0);

    let ngram_count = count_ngram_occurrences(ngram_cooccurrences.clone(), ngram);
    let label_count = count_label_occurrences(ngram_cooccurrences.clone(), label);

    let p_ngram_label = (ngram_label_count as f64 + smoothing)
        / (total_count as f64 + smoothing * vocab_size as f64);
    let p_ngram =
        (ngram_count as f64 + smoothing) / (total_count as f64 + smoothing * vocab_size as f64);
    let p_label =
        (label_count as f64 + smoothing) / (total_count as f64 + smoothing * vocab_size as f64);

    let pmi = p_ngram_label / (p_ngram * p_label);
    Float::max(pmi, 0.0)
}

#[pyfunction]
fn compute_pmi_for_all_ngrams(
    ngram_cooccurrences: HashMap<(String, String), usize>,
    ngrams: Vec<String>,
    label: &str,
    vocab_size: usize,
    total_count: usize,
    smoothing: f64,
) -> Vec<f64> {
    ngrams
        .par_iter()
        .progress_count(ngrams.len() as u64)
        .map(|ngram| {
            compute_pmi(
                ngram_cooccurrences.clone(),
                ngram,
                label,
                vocab_size,
                total_count,
                smoothing,
            )
        })
        .collect()
}

#[pyfunction]
fn compute_pmi_for_all_keys(
    ngram_cooccurrences: HashMap<(String, String), usize>,
    vocab_size: usize,
    total_count: usize,
    smoothing: f64,
) -> HashMap<(String, String), f64> {
    let keys: Vec<(String, String)> = ngram_cooccurrences.keys().cloned().collect();

    keys.par_iter()
        .progress_count(keys.len() as u64)
        .map(|(ngram, label)| {
            let pmi = compute_pmi(
                ngram_cooccurrences.clone(),
                ngram,
                label,
                vocab_size,
                total_count,
                smoothing,
            );
            ((ngram.clone(), label.clone()), pmi)
        })
        .collect()
}

// Applicability utils

#[pyfunction]
fn count_all_ngram_instance_occurrences(raw_dataset: &str) -> PyResult<HashMap<String, usize>> {
    let dataset: Vec<DatasetItem> = serde_json::from_str(raw_dataset).map_err(SerdeJsonError)?;

    let mut ngram_instance_count: HashMap<String, usize> = HashMap::new();

    for item in dataset {
        let mut ngrams_seen: HashSet<String> = HashSet::new();
        for choice in item.choices {
            for ngram in choice.ngrams {
                if !ngrams_seen.contains(&ngram) {
                    *ngram_instance_count.entry(ngram.clone()).or_insert(0) += 1;
                    ngrams_seen.insert(ngram);
                }
            }
        }
    }

    Ok(ngram_instance_count)
}

#[pyfunction]
fn get_nivens_applicability(raw_dataset: &str) -> PyResult<HashMap<String, usize>> {
    let dataset: Vec<DatasetItem> = serde_json::from_str(raw_dataset).map_err(SerdeJsonError)?;

    // applicable ngrams occur in only one choice and are not present in any other choice
    let mut applicable_ngrams: HashMap<String, usize> = HashMap::new();

    for item in dataset.into_iter().progress() {
        let correct_answer_ngrams: HashSet<String> = item
            .choices
            .iter()
            .filter(|choice| choice.label == true)
            .flat_map(|choice| choice.ngrams.clone())
            .collect();

        let wrong_answer_ngrams: HashSet<String> = item
            .choices
            .iter()
            .filter(|choice| choice.label == false)
            .flat_map(|choice| choice.ngrams.clone())
            .collect();

        let exclusive_correct_answer_ngrams = correct_answer_ngrams
            .difference(&wrong_answer_ngrams)
            .cloned()
            .collect::<Vec<String>>();

        let exclusive_wrong_answer_ngrams = wrong_answer_ngrams
            .difference(&correct_answer_ngrams)
            .cloned()
            .collect::<Vec<String>>();

        for ngram in exclusive_correct_answer_ngrams {
            *applicable_ngrams.entry(ngram).or_insert(0) += 1;
        }

        for ngram in exclusive_wrong_answer_ngrams {
            *applicable_ngrams.entry(ngram).or_insert(0) += 1;
        }
    }

    Ok(applicable_ngrams)
}

#[pyfunction]
fn count_unique_ngrams_in_correct_and_wrong_answers(
    raw_dataset: &str,
) -> PyResult<(HashMap<String, usize>, HashMap<String, usize>)> {
    let dataset: Vec<DatasetItem> = serde_json::from_str(raw_dataset).map_err(SerdeJsonError)?;

    let mut correct_answer_unique_ngrams: HashMap<String, usize> = HashMap::new();
    let mut wrong_answer_unique_ngrams: HashMap<String, usize> = HashMap::new();

    for item in dataset {
        let correct_answer_ngrams: HashSet<String> = item
            .choices
            .iter()
            .filter(|choice| choice.label == true)
            .flat_map(|choice| choice.ngrams.clone())
            .collect();

        let wrong_answer_ngrams: HashSet<String> = item
            .choices
            .iter()
            .filter(|choice| choice.label == false)
            .flat_map(|choice| choice.ngrams.clone())
            .collect();

        let unique_correct_answer_ngrams = correct_answer_ngrams
            .difference(&wrong_answer_ngrams)
            .cloned()
            .collect::<Vec<String>>();

        let unique_wrong_answer_ngrams = wrong_answer_ngrams
            .difference(&correct_answer_ngrams)
            .cloned()
            .collect::<Vec<String>>();

        for ngram in unique_correct_answer_ngrams {
            *correct_answer_unique_ngrams.entry(ngram).or_insert(0) += 1;
        }

        for ngram in unique_wrong_answer_ngrams {
            *wrong_answer_unique_ngrams.entry(ngram).or_insert(0) += 1;
        }
    }

    Ok((correct_answer_unique_ngrams, wrong_answer_unique_ngrams))
}

/// A Python module implemented in Rust.
#[pymodule]
fn scue(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(get_all_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(get_n_from_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(get_n_from_ngram_list, m)?)?;
    m.add_function(wrap_pyfunction!(count_ngram_cooccurrences, m)?)?;
    m.add_function(wrap_pyfunction!(fast_count_ngram_cooccurrences, m)?)?;
    m.add_function(wrap_pyfunction!(count_ngram_occurrences, m)?)?;
    m.add_function(wrap_pyfunction!(count_label_occurrences, m)?)?;
    m.add_function(wrap_pyfunction!(get_vocab_size_total_counts, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pmi, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pmi_for_all_ngrams, m)?)?;
    m.add_function(wrap_pyfunction!(compute_pmi_for_all_keys, m)?)?;
    m.add_function(wrap_pyfunction!(count_all_ngram_instance_occurrences, m)?)?;
    m.add_function(wrap_pyfunction!(
        count_unique_ngrams_in_correct_and_wrong_answers,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(get_nivens_applicability, m)?)?;
    Ok(())
}
