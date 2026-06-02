use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[tokio::test(flavor = "multi_thread")]
async fn test_capability_link_py() {
    Python::attach(|py| {
        let module = iris::py::make_module(py).expect("make_module");
        let rt = module.getattr("PyRuntime").unwrap().call0().unwrap();

        let locals = PyDict::new(py);
        locals.set_item("rt", rt.clone()).unwrap();

        // 1. Spawn an actor and get a link
        py.run(
            c_str!(
                r#"
def handler(msg):
    # Push-based handler receives the message directly
    pass
"#
            ),
            None,
            Some(&locals),
        )
        .unwrap();

        let handler = locals.get_item("handler").unwrap().unwrap();

        let link = rt.call_method1("spawn_link", (handler,)).unwrap();

        // 2. Try to send via link
        let res = rt
            .call_method1("send_link", (link.clone(), b"hello"))
            .unwrap();
        assert!(res.extract::<bool>().unwrap());

        // 3. Restrict link and try to send (should fail if Send cap removed)
        let caps_mod = module.getattr("PyCapability").unwrap();
        let send_cap = caps_mod.getattr("Send").unwrap();
        let signal_cap = caps_mod.getattr("Signal").unwrap();

        let restricted_link = link.call_method1("restrict", (vec![signal_cap],)).unwrap();
        let res2 = rt
            .call_method1("send_link", (restricted_link, b"world"))
            .unwrap();
        assert!(!res2.extract::<bool>().unwrap());
    });
}
