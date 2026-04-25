#![cfg(feature = "pyo3")]

use pyo3::prelude::*;

#[tokio::test]
async fn test_py_call_response() {
    let result: Vec<u8> = Python::attach(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let runtime_type = module.getattr("PyRuntime").expect("no PyRuntime type");
        let rt_obj = runtime_type.call0().expect("construct PyRuntime");

        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("rt", rt_obj).unwrap();
        locals
            .set_item("__builtins__", py.import("builtins").unwrap())
            .unwrap();

        py.run(
            pyo3::ffi::c_str!(
                r#"
import asyncio

def responder(msg, rt=rt):
    if hasattr(msg, 'reply_to'):
        rt.send(msg.reply_to, b"pong")

pid = rt.spawn_py_handler(responder, 10)

async def run_call():
    # rt.call returns a coroutine
    coro = rt.call(pid, b"ping", 1.0)
    return await coro

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
result = loop.run_until_complete(run_call())
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();

        let res = locals.get_item("result").unwrap().unwrap();
        res.extract::<Vec<u8>>()
    })
    .unwrap();

    assert_eq!(result, b"pong");
}
