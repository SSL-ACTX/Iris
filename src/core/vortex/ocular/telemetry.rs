use crate::vortex::ocular::model::{CodeMeta, TraceEvent, TraceStats};
use crate::vortex::ocular::state::{
    get_exclude_patterns, get_include_patterns, set_observed_offsets_by_code, set_summary,
    TraceSummary, DEINSTRUMENT_THRESHOLD, EVENT_QUEUE, FREE_QUEUE, IS_PRECISE, IS_RUNNING,
    PROCESSED_EVENTS,
};
use pyo3::prelude::*;

const HOT_TRACE_EXPORT: bool = false;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::Ordering;
use std::thread;
use std::time::Duration;

pub fn telemetry_worker() {
    let queue = EVENT_QUEUE
        .get()
        .expect("EVENT_QUEUE must be initialized before worker starts");
    let mut processed_events: u64 = 0;

    let mut code_registry: HashMap<usize, CodeMeta> = HashMap::new();
    let mut code_hot_hits: HashMap<usize, u64> = HashMap::new();
    let mut call_stack: Vec<(usize, i32, u64)> = Vec::with_capacity(64);

    let mut current_trace: Vec<(usize, i32, u64)> = Vec::with_capacity(256);
    let mut current_trace_cycles: u64 = 0;

    let mut hot_traces: HashMap<Vec<(usize, i32)>, TraceStats> = HashMap::new();
    let mut instruction_hits: HashMap<(usize, i32), u64> = HashMap::new();
    let mut instruction_events: u64 = 0;

    while IS_RUNNING.load(Ordering::Relaxed) || !queue.is_empty() {
        if let Some(mut batch) = queue.pop() {
            for event in batch.drain(..) {
                processed_events += 1;
                PROCESSED_EVENTS.store(processed_events, Ordering::Relaxed);

                match event {
                    TraceEvent::PyStart {
                        code,
                        code_ptr,
                        lasti,
                        tsc,
                        ..
                    } => {
                        if let Some(code_obj) = code {
                            if let std::collections::hash_map::Entry::Vacant(e) =
                                code_registry.entry(code_ptr)
                            {
                                Python::attach(|py| {
                                    let bound_code = code_obj.bind(py);

                                    let func_name = bound_code
                                        .getattr("co_name")
                                        .and_then(|n| n.extract::<String>())
                                        .unwrap_or_else(|_| "unknown".to_string());

                                    let mut base_opcodes = HashMap::new();
                                    let mut valid_offsets = Vec::new();

                                    if let Ok(dis) = py.import(pyo3::ffi::c_str!("dis")) {
                                        let kwargs = pyo3::types::PyDict::new(py);
                                        let _ = kwargs.set_item("adaptive", false);

                                        if let Ok(instructions) = dis.call_method(
                                            "get_instructions",
                                            (bound_code,),
                                            Some(&kwargs),
                                        ) {
                                            if let Ok(iter) = instructions.try_iter() {
                                                for inst_res in iter {
                                                    let inst: Bound<'_, PyAny> = inst_res?;
                                                    let offset = inst
                                                        .getattr("offset")
                                                        .ok()
                                                        .and_then(|o| o.extract::<i32>().ok());
                                                    let opname = inst
                                                        .getattr("opname")
                                                        .ok()
                                                        .and_then(|o| o.extract::<String>().ok());
                                                    let arg = inst
                                                        .getattr("arg")
                                                        .ok()
                                                        .and_then(|o| o.extract::<i32>().ok());
                                                    let argrepr = inst
                                                        .getattr("argrepr")
                                                        .ok()
                                                        .and_then(|o| o.extract::<String>().ok());
                                                    let starts_line = inst
                                                        .getattr("starts_line")
                                                        .ok()
                                                        .and_then(|o| o.extract::<i32>().ok());
                                                    let is_jump_target = inst
                                                        .getattr("is_jump_target")
                                                        .ok()
                                                        .and_then(|o| o.extract::<bool>().ok())
                                                        .unwrap_or(false);

                                                    if let (Some(off), Some(name)) =
                                                        (offset, opname)
                                                    {
                                                        base_opcodes.insert(
                                                            off,
                                                            crate::vortex::ocular::model::InstMeta {
                                                                opname: name,
                                                                arg,
                                                                argrepr,
                                                                starts_line,
                                                                is_jump_target,
                                                            },
                                                        );
                                                        valid_offsets.push(off);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    valid_offsets.sort();

                                    let filename = bound_code
                                        .getattr("co_filename")
                                        .and_then(|f| f.extract::<String>())
                                        .unwrap_or_else(|_| "<unknown>".to_string());
                                    let firstlineno = bound_code
                                        .getattr("co_firstlineno")
                                        .and_then(|f| f.extract::<i32>())
                                        .unwrap_or(-1);

                                    e.insert(CodeMeta {
                                        name: func_name,
                                        code_obj: code_obj.clone_ref(py),
                                        base_opcodes,
                                        valid_offsets,
                                        filename,
                                        firstlineno,
                                    });
                                    Ok::<(), PyErr>(())
                                })
                                .unwrap();
                            }
                        }

                        call_stack.push((code_ptr, lasti, tsc));
                    }
                    TraceEvent::Instruction {
                        code_ptr,
                        lasti,
                        tsc,
                    } => {
                        instruction_events = instruction_events.saturating_add(1);
                        *instruction_hits.entry((code_ptr, lasti)).or_insert(0) += 1;
                        current_trace.push((code_ptr, lasti, tsc));
                    }
                    TraceEvent::Jump {
                        code_ptr,
                        from_lasti,
                        to_lasti,
                        tsc,
                        ..
                    } => {
                        if let Some(top) = call_stack.last_mut() {
                            if top.0 == code_ptr {
                                let start_pc = top.1;
                                let last_tsc = top.2;
                                let is_precise = IS_PRECISE.load(Ordering::Relaxed);

                                if !is_precise {
                                    if let Some(meta) = code_registry.get(&code_ptr) {
                                        for &offset in &meta.valid_offsets {
                                            if offset >= start_pc && offset <= from_lasti {
                                                current_trace.push((code_ptr, offset, 0));
                                            }
                                        }
                                    }
                                }

                                let block_cycles = tsc.saturating_sub(last_tsc);
                                current_trace_cycles += block_cycles;

                                top.1 = to_lasti;
                                top.2 = tsc;

                                if to_lasti < from_lasti && !current_trace.is_empty() {
                                    let trace_key: Vec<(usize, i32)> =
                                        current_trace.iter().map(|&(c, l, _)| (c, l)).collect();
                                    let len = current_trace.len();

                                    let stats =
                                        hot_traces.entry(trace_key).or_insert_with(|| TraceStats {
                                            hits: 0,
                                            cycles: vec![0; len],
                                        });
                                    stats.hits += 1;

                                    if !is_precise {
                                        let avg = if len > 0 {
                                            current_trace_cycles / len as u64
                                        } else {
                                            0
                                        };
                                        for i in 0..len {
                                            stats.cycles[i] += avg;
                                        }
                                    } else {
                                        for i in 0..len {
                                            let current_tsc = current_trace[i].2;
                                            let next_tsc = if i + 1 < len {
                                                current_trace[i + 1].2
                                            } else {
                                                tsc
                                            };
                                            stats.cycles[i] += next_tsc.saturating_sub(current_tsc);
                                        }
                                    }

                                    *code_hot_hits.entry(code_ptr).or_insert(0) += 1;
                                    current_trace.clear();
                                    current_trace_cycles = 0;
                                }
                            }
                        }
                    }
                    TraceEvent::PyReturn { code_ptr, .. } => {
                        if let Some((top_code_ptr, _, _)) = call_stack.last() {
                            if *top_code_ptr == code_ptr {
                                call_stack.pop();
                            } else {
                                while let Some(top) = call_stack.last() {
                                    if top.0 == code_ptr {
                                        call_stack.pop();
                                        break;
                                    }
                                    call_stack.pop();
                                }
                            }
                        }

                        current_trace.clear();
                        current_trace_cycles = 0;
                    }
                }
            }

            if let Some(free_q) = FREE_QUEUE.get() {
                let _ = free_q.push(batch);
            }
        } else if IS_RUNNING.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(1));
        }
    }

    println!(
        "[Ocular] Telemetry worker gracefully exited. Processed {} events.",
        processed_events
    );

    let mode_label = if IS_PRECISE.load(Ordering::Relaxed) {
        "precise"
    } else {
        "adaptive"
    };
    let deinstrument_threshold = DEINSTRUMENT_THRESHOLD.load(Ordering::Relaxed);
    let exclude_patterns = get_exclude_patterns();
    let include_patterns = get_include_patterns();

    println!("[Ocular] =================================================");
    println!("[Ocular] Ocular Telemetry UI");
    println!("[Ocular] mode                 = {}", mode_label);
    println!(
        "[Ocular] deinstrument_threshold = {}",
        deinstrument_threshold
    );
    if !exclude_patterns.is_empty() {
        println!("[Ocular] exclude patterns      = {:?}", exclude_patterns);
    }
    if !include_patterns.is_empty() {
        println!("[Ocular] include_only patterns = {:?}", include_patterns);
    }
    println!("[Ocular] =================================================");

    if !instruction_hits.is_empty() {
        let mut top_hits: Vec<((usize, i32), u64)> =
            instruction_hits.iter().map(|(k, v)| (*k, *v)).collect();
        top_hits.sort_by(|a, b| b.1.cmp(&a.1));

        println!("[Ocular] ------------------------------------------------");
        println!("[Ocular] Top Instruction Sites (all captured activity):");
        for ((code_ptr, offset), hits) in top_hits.into_iter().take(12) {
            let opname = code_registry
                .get(&code_ptr)
                .and_then(|m| m.base_opcodes.get(&offset))
                .map(|m| m.opname.clone())
                .unwrap_or_else(|| "UNKNOWN".to_string());
            println!(
                "[Ocular] code=0x{:016x} offset={:<4} op={:<24} hits={}",
                code_ptr, offset, opname, hits
            );
        }
    }

    if let Some((top_trace, stats)) = hot_traces.into_iter().max_by_key(|entry| entry.1.hits) {
        println!("[Ocular] ------------------------------------------------");
        println!("[Ocular] Top Hot Trace Detected:");
        println!("[Ocular] Hot trace length : {} uOps", top_trace.len());
        println!("[Ocular] Hits:   {}", stats.hits);

        let mut trace_dump_lines = Vec::new();
        trace_dump_lines.push("Ocular Hot Trace Dump".to_string());
        trace_dump_lines.push(format!("Length: {} uOps", top_trace.len()));
        trace_dump_lines.push(format!("Hits:   {}", stats.hits));
        trace_dump_lines.push("------------------------------------------------".to_string());

        let is_precise = IS_PRECISE.load(Ordering::Relaxed);
        Python::attach(|py| {
            let mut disassembly_cache: HashMap<usize, HashMap<i32, String>> = HashMap::new();
            let dis_module = py.import(pyo3::ffi::c_str!("dis")).ok();

            for (idx, (c_ptr, lasti)) in top_trace.into_iter().enumerate() {
                let mut opcode_base = "UNKNOWN".to_string();
                let mut opcode_quickened = "UNKNOWN".to_string();

                if let Some(meta) = code_registry.get(&c_ptr) {
                    if let Some(inst_meta) = meta.base_opcodes.get(&lasti) {
                        opcode_base = inst_meta.opname.clone();
                    }

                    let inst_map = disassembly_cache.entry(c_ptr).or_insert_with(|| {
                        let mut map = HashMap::new();
                        if let Some(dis) = &dis_module {
                            let kwargs = pyo3::types::PyDict::new(py);
                            let _ = kwargs.set_item("adaptive", true);
                            let mut dis_result = dis.call_method(
                                "get_instructions",
                                (meta.code_obj.bind(py),),
                                Some(&kwargs),
                            );

                            if dis_result.is_err() {
                                let fallback_kwargs = pyo3::types::PyDict::new(py);
                                let _ = fallback_kwargs.set_item("adaptive", false);
                                dis_result = dis.call_method(
                                    "get_instructions",
                                    (meta.code_obj.bind(py),),
                                    Some(&fallback_kwargs),
                                );
                            }

                            if let Ok(instructions) = dis_result {
                                if let Ok(iter) = instructions.try_iter() {
                                    for inst in iter.flatten() {
                                        let offset = inst
                                            .getattr("offset")
                                            .ok()
                                            .and_then(|o| o.extract::<i32>().ok());
                                        let opname = inst
                                            .getattr("opname")
                                            .ok()
                                            .and_then(|o| o.extract::<String>().ok());
                                        if let (Some(off), Some(name)) = (offset, opname) {
                                            map.insert(off, name);
                                        }
                                    }
                                }
                            }
                        }
                        map
                    });

                    opcode_quickened = inst_map
                        .get(&lasti)
                        .cloned()
                        .map(|n| n.replace("INSTRUMENTED_", ""))
                        .unwrap_or_else(|| opcode_base.clone());
                }

                let avg_cycles = stats.cycles[idx] / stats.hits;
                let transition = if opcode_base == opcode_quickened {
                    opcode_base
                } else {
                    format!("{} -> {}", opcode_base, opcode_quickened)
                };

                let line = if !is_precise {
                    format!(
                        "[Ocular] {:>3} 0x{:016x} {:<40} | ~{} cycles (avg block latency)",
                        idx, c_ptr, transition, avg_cycles
                    )
                } else {
                    format!(
                        "[Ocular] {:>3} 0x{:016x} {:<40} | ~{} cycles",
                        idx, c_ptr, transition, avg_cycles
                    )
                };

                println!("{}", line);
                trace_dump_lines.push(line);
            }
            Ok::<(), PyErr>(())
        })
        .unwrap();

        println!("[Ocular] ------------------------------------------------");

        if HOT_TRACE_EXPORT {
            if let Ok(file) = File::create("ocular_hot_trace.txt") {
                let mut writer = BufWriter::new(file);
                for line in trace_dump_lines {
                    let _ = writeln!(writer, "{}", line);
                }
                println!("[Ocular] Hot trace disassembly exported to 'ocular_hot_trace.txt'");
            }
        }
    } else {
        println!("[Ocular] No hot traces (loops) detected.");
    }

    set_summary(TraceSummary {
        processed_events,
        instruction_events,
        unique_instruction_sites: instruction_hits.len(),
        loop_trace_count: code_hot_hits.values().copied().sum::<u64>() as usize,
    });

    let mut observed_by_code: HashMap<usize, Vec<(i32, u64)>> = HashMap::new();
    for ((code_ptr, offset), hits) in instruction_hits {
        observed_by_code
            .entry(code_ptr)
            .or_default()
            .push((offset, hits));
    }
    let compact_map: HashMap<usize, Vec<i32>> = observed_by_code
        .into_iter()
        .map(|(code_ptr, mut rows)| {
            rows.sort_by(|a, b| b.1.cmp(&a.1));
            let offsets = rows
                .into_iter()
                .map(|(offset, _)| offset)
                .collect::<Vec<_>>();
            (code_ptr, offsets)
        })
        .collect();
    set_observed_offsets_by_code(compact_map);
}
