# Copilot instructions — Project Iris (core runtime)

Purpose
- Give AI coding agents the exact, actionable knowledge to be productive immediately in this Rust core runtime (Phase 1).
- Enforce **Strict TDD**: write a failing test first, implement, then refactor. CI requires all tests to pass.

Big picture (read these files first)
- Architecture: lightweight actor runtime implemented in Rust (no FFI in Phase 1).
  - `src/pid.rs` — generational SlabAllocator; `Pid` is `u64` (generation<<32 | index). Do NOT change PID representation without updating all callers and FFI plans.
  - `src/mailbox.rs` — mailbox primitive; `Message` = `Vec<u8>` and uses `tokio::mpsc::unbounded_channel()`.
  - `src/supervisor.rs` — supervision stub (restart strategies are TODO; tests expect watch/unwatch behavior).
  - `src/lib.rs` — `Runtime` API (`new()`, `spawn_actor(...)`, `send(pid, msg)`, `is_alive`).

Key design decisions & why
- PIDs are integer IDs (u64 generational slab) to make FFI/serialization safe and avoid dangling pointers.
- Mailboxes carry binary blobs (`Vec<u8>`) so serialization strategy can be swapped later (serde/msgpack/Arrow).
- Cooperative preemption implemented at Future wrapper level (reduction counter) to keep actor fairness without a custom OS scheduler.
- Concurrency uses `dashmap` and `Arc/Mutex` for fast lookups and safe sharing across tokio worker threads.

Developer workflows (essential commands)
- Build & test: `cargo test` (CI runs the same). Run entire workspace with `cargo test --manifest-path ./Cargo.toml`.
- Run example: `cargo run --example basic`.
- Format: `cargo fmt` (project has `rustfmt.toml`).
- CI: `.github/workflows/ci.yml` runs `cargo test` on push/PR.

Project conventions & patterns
- Strict TDD: add unit test in the same module (`#[cfg(test)]`) or `tests/*.rs` before implementing behavior.
- Async tests use `#[tokio::test]`.
- Public API changes must update `README.md` examples and add tests.
- When adding/altering message envelopes, update `src/mailbox.rs` and all `send`/`recv` callsites; tests rely on `send` returning `Result<(), Message>` where a failure returns the original payload.
- Use `DashMap` for concurrent maps (see `Runtime.mailboxes`).

Integration points / future expansions
- PyO3 and `napi-rs` membranes are planned (Phase 2+). Keep `Pid` as `u64` and `Message` as a binary envelope to simplify FFI.
- Serialization roadmap: `serde` + `rmp-serde` (MessagePack) by convention.

Where to look for examples/tests
- Unit tests: inline in modules (`src/*.rs`) and `#[cfg(test)]` blocks.
- Integration tests: `tests/*.rs` (e.g. `tests/integration_test.rs`).
- Example runtime usage: `examples/basic.rs`.

Files to touch for common tasks
- Add actor behavior / scheduler work: `src/lib.rs`.
- Add mailbox/envelope changes: `src/mailbox.rs` + update tests in `mailbox` and `integration_test.rs`.
- Add supervision strategies: `src/supervisor.rs` + unit + integration tests.

Rules for AI contributors (must follow)
1. Follow Strict TDD: produce failing tests first; CI must be green before PR. ✅
2. Do not change `Pid` layout or `Message` type without adding migration/tests. ⚠️
3. Preserve concurrency invariants: prefer `DashMap`/`Arc` and avoid global mutable state. 🔧
4. Use `tokio` async patterns and `#[tokio::test]` for async behavior. 🔁
5. Apply **DRY** principles: refactor duplicate logic into shared helpers and avoid copy-paste. 💡
6. When modifying `ReductionLimiter` or other pinned futures, demonstrate correctness with a unit test covering yield behavior.
