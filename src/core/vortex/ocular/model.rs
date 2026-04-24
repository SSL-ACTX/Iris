use pyo3::prelude::*;
use std::collections::HashMap;

#[derive(Debug)]
#[allow(dead_code)]
pub enum TraceEvent {
    PyStart {
        code: Option<PyObject>,
        code_ptr: usize,
        lasti: i32,
        ts: u64,
        tsc: u64,
    },
    Instruction {
        code_ptr: usize,
        lasti: i32,
        tsc: u64,
    },
    Jump {
        code_ptr: usize,
        from_lasti: i32,
        to_lasti: i32,
        ts: u64,
        tsc: u64,
    },
    PyReturn {
        code_ptr: usize,
        ts: u64,
        tsc: u64,
    },
}

#[derive(Clone)]
pub struct InstMeta {
    pub opname: String,
    pub arg: Option<i32>,
    pub argrepr: Option<String>,
    pub starts_line: Option<i32>,
    pub is_jump_target: bool,
}

pub struct CodeMeta {
    pub name: String,
    pub code_obj: PyObject,
    pub base_opcodes: HashMap<i32, InstMeta>,
    pub valid_offsets: Vec<i32>,
    pub filename: String,
    pub firstlineno: i32,
}

pub struct TraceStats {
    pub hits: u64,
    pub cycles: Vec<u64>,
}
