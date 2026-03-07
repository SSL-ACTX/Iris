import unittest
import array

import iris.jit as jit_mod


class TestJitFallback(unittest.TestCase):
    def test_jit_panic_runtime_error_falls_back(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        try:
            def fake_call_jit(_func, _args, _kwargs):
                raise RuntimeError("jit panic: simulated")

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = lambda *a, **k: None

            @jit_mod.offload(strategy="jit")
            def add1(x):
                return x + 1

            self.assertEqual(add1(41), 42)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_complex_body_skips_jit_wrapper(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        try:
            def fake_call_jit(_func, _args, _kwargs):
                raise AssertionError("call_jit should not run for complex function body")

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = lambda *a, **k: None

            @jit_mod.offload(strategy="jit")
            def prng_like(seed, n):
                numbers = []
                x = seed
                for i in range(n):
                    if i % 2 == 0:
                        x += 0.1
                    x = (x + i * 0.1) % 1.0
                    numbers.append(x)
                return numbers

            out = prng_like(0.25, 3)
            self.assertEqual(len(out), 3)
            self.assertAlmostEqual(out[0], 0.35)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_stateful_loop_uses_step_jit(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        calls = {"jit": 0, "register": 0}

        try:
            def fake_call_jit(_func, _args, _kwargs):
                calls["jit"] += 1
                return _func(*_args)

            def fake_register_offload(*_args, **_kwargs):
                calls["register"] += 1
                return None

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = fake_register_offload

            @jit_mod.offload(strategy="jit")
            def accum(seed, n):
                numbers = []
                x = seed
                for i in range(n):
                    x = x + i + 1
                    numbers.append(x)
                return numbers

            out = accum(0.0, 3)
            self.assertEqual(out, [1.0, 3.0, 6.0])
            self.assertGreaterEqual(calls["register"], 1)
            self.assertEqual(calls["jit"], 3)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_complex_body_array_inputs_vectorized_fallback(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        try:
            jit_mod.call_jit = lambda _func, _args, _kwargs: (_ for _ in ()).throw(
                RuntimeError("no JIT entry found")
            )
            jit_mod.register_offload = lambda *a, **k: None

            @jit_mod.offload(strategy="jit", return_type="float")
            def branch_like(price, vol, strike):
                x = price / strike
                return x + vol

            prices = array.array("d", [100.0, 101.0, 102.0])
            vols = array.array("d", [0.2, 0.2, 0.2])
            strikes = array.array("d", [105.0, 105.0, 105.0])

            out = branch_like(prices, vols, strikes)
            self.assertEqual(len(out), 3)
            self.assertAlmostEqual(float(out[0]), (100.0 / 105.0) + 0.2)
            self.assertAlmostEqual(float(out[2]), (102.0 / 105.0) + 0.2)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_complex_body_vector_inputs_use_aggressive_jit(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        calls = {"jit": 0, "register": 0}

        try:
            def fake_call_jit(_func, _args, _kwargs):
                calls["jit"] += 1
                if isinstance(_args[0], (int, float)):
                    return (_args[0] / _args[2]) + _args[1]
                # mimic old aggressive mode behavior for vector path
                return _args[0]

            def fake_register_offload(*_args, **_kwargs):
                calls["register"] += 1
                return None

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = fake_register_offload

            @jit_mod.offload(strategy="jit", return_type="float")
            def branch_like(price, vol, strike):
                x = price / strike
                return x + vol

            scalar_out = branch_like(100.0, 0.2, 105.0)
            self.assertAlmostEqual(float(scalar_out), (100.0 / 105.0) + 0.2)

            prices = array.array("d", [100.0, 101.0, 102.0])
            vols = array.array("d", [0.2, 0.2, 0.2])
            strikes = array.array("d", [105.0, 105.0, 105.0])
            out = branch_like(prices, vols, strikes)
            self.assertEqual(len(out), 3)
            self.assertGreaterEqual(calls["register"], 1)
            self.assertEqual(calls["jit"], 2)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_inlined_assignments_use_scalar_jit_path(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        calls = {"jit": 0, "register": 0}
        seen = {"src": None}

        try:
            def fake_call_jit(_func, _args, _kwargs):
                calls["jit"] += 1
                return 123.0

            def fake_register_offload(_func, _strategy, _return_type, src, _arg_names):
                calls["register"] += 1
                seen["src"] = src
                return None

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = fake_register_offload

            @jit_mod.offload(strategy="jit", return_type="float")
            def calc(price, strike, vol):
                x = price / strike
                y = x + vol
                return y * 2

            out = calc(100.0, 105.0, 0.2)
            self.assertEqual(out, 123.0)
            self.assertGreaterEqual(calls["register"], 1)
            self.assertEqual(calls["jit"], 1)
            self.assertIsNotNone(seen["src"])

        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_inlined_if_else_use_scalar_jit_path(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        calls = {"jit": 0}

        try:
            def fake_call_jit(_func, _args, _kwargs):
                calls["jit"] += 1
                return 7.0

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = lambda *a, **k: None

            @jit_mod.offload(strategy="jit", return_type="float")
            def branchy(x, y):
                z = x - y
                if z > 0:
                    out = z * 2
                else:
                    out = y - x
                return out + 1

            out = branchy(3.0, 2.0)
            self.assertEqual(out, 7.0)
            self.assertEqual(calls["jit"], 1)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_inlined_elif_chain_use_scalar_jit_path(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        calls = {"jit": 0}

        try:
            def fake_call_jit(_func, _args, _kwargs):
                calls["jit"] += 1
                return 11.0

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = lambda *a, **k: None

            @jit_mod.offload(strategy="jit", return_type="float")
            def branchy_elif(x, y):
                z = x - y
                if z > 3:
                    out = z * 2
                elif z > 0:
                    out = z + 4
                else:
                    out = y - x
                return out + 1

            out = branchy_elif(5.0, 2.0)
            self.assertEqual(out, 11.0)
            self.assertEqual(calls["jit"], 1)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_inlined_if_without_else_use_scalar_jit_path(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        calls = {"jit": 0}

        try:
            def fake_call_jit(_func, _args, _kwargs):
                calls["jit"] += 1
                return 9.0

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = lambda *a, **k: None

            @jit_mod.offload(strategy="jit", return_type="float")
            def branchy_if_only(x, y):
                out = x
                if x > y:
                    out = out + 2
                return out + 1

            out = branchy_if_only(4.0, 3.0)
            self.assertEqual(out, 9.0)
            self.assertEqual(calls["jit"], 1)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_inlined_pow_builtin_normalized(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        seen = {"src": None}

        try:
            jit_mod.call_jit = lambda _func, _args, _kwargs: 5.0

            def fake_register_offload(_func, _strategy, _return_type, src, _arg_names):
                seen["src"] = src
                return None

            jit_mod.register_offload = fake_register_offload

            @jit_mod.offload(strategy="jit", return_type="float")
            def uses_pow(x, y):
                z = pow(x, y)
                return z + 1

            _ = uses_pow(2.0, 3.0)
            self.assertIsNotNone(seen["src"])
            self.assertIn("**", str(seen["src"]))
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_inlined_annassign_use_scalar_jit_path(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        calls = {"jit": 0}

        try:
            def fake_call_jit(_func, _args, _kwargs):
                calls["jit"] += 1
                return 21.0

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = lambda *a, **k: None

            @jit_mod.offload(strategy="jit", return_type="float")
            def ann_calc(x: float, y: float):
                z: float = x + y
                return z * 2

            out = ann_calc(2.0, 3.0)
            self.assertEqual(out, 21.0)
            self.assertEqual(calls["jit"], 1)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_scalar_while_uses_step_jit(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload
        original_step_loop = jit_mod.call_jit_step_loop_f64

        calls = {"jit": 0, "register": 0}

        try:
            def fake_call_jit(_func, _args, _kwargs):
                calls["jit"] += 1
                return _func(*_args)

            def fake_register_offload(*_args, **_kwargs):
                calls["register"] += 1
                return None

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = fake_register_offload
            jit_mod.call_jit_step_loop_f64 = None

            @jit_mod.offload(strategy="jit", return_type="float")
            def scalar_loop(seed, n):
                x = seed
                i = 0
                while i < n:
                    x = x + i + 1
                    i += 1
                return x

            out = scalar_loop(0.0, 3)
            self.assertEqual(out, 6.0)
            self.assertGreaterEqual(calls["register"], 1)
            self.assertEqual(calls["jit"], 3)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload
            jit_mod.call_jit_step_loop_f64 = original_step_loop

    def test_scalar_for_uses_step_jit(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload
        original_step_loop = jit_mod.call_jit_step_loop_f64

        calls = {"jit": 0, "register": 0}

        try:
            def fake_call_jit(_func, _args, _kwargs):
                calls["jit"] += 1
                return _func(*_args)

            def fake_register_offload(*_args, **_kwargs):
                calls["register"] += 1
                return None

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = fake_register_offload
            jit_mod.call_jit_step_loop_f64 = None

            @jit_mod.offload(strategy="jit", return_type="float")
            def scalar_for(seed, n):
                x = seed
                for i in range(n):
                    x = x + i + 1
                return x

            out = scalar_for(0.0, 3)
            self.assertEqual(out, 6.0)
            self.assertGreaterEqual(calls["register"], 1)
            self.assertEqual(calls["jit"], 3)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload
            jit_mod.call_jit_step_loop_f64 = original_step_loop

    def test_scalar_for_prefers_rust_step_loop_api(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload
        original_step_loop = jit_mod.call_jit_step_loop_f64

        calls = {"step_loop": 0}

        try:
            jit_mod.call_jit = lambda _func, _args, _kwargs: (_ for _ in ()).throw(
                AssertionError("per-iteration call_jit should not be used when step loop API is available")
            )
            jit_mod.register_offload = lambda *a, **k: None

            def fake_step_loop(step_fn, seed, count):
                calls["step_loop"] += 1
                state = float(seed)
                for i in range(int(count)):
                    state = float(step_fn(state, float(i)))
                return state

            jit_mod.call_jit_step_loop_f64 = fake_step_loop

            @jit_mod.offload(strategy="jit", return_type="float")
            def scalar_for(seed, n):
                x = seed
                for i in range(n):
                    x = x + i + 1
                return x

            out = scalar_for(0.0, 3)
            self.assertEqual(out, 6.0)
            self.assertEqual(calls["step_loop"], 1)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload
            jit_mod.call_jit_step_loop_f64 = original_step_loop

    def test_scalar_for_vector_inputs_fallback(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload

        try:
            jit_mod.call_jit = lambda _func, _args, _kwargs: (_ for _ in ()).throw(
                AssertionError("scalar-for wrapper should use vectorized python fallback for vector inputs")
            )
            jit_mod.register_offload = lambda *a, **k: None

            @jit_mod.offload(strategy="jit", return_type="float")
            def scalar_for(seed, n):
                x = seed
                for i in range(n):
                    x = x + i + 1
                return x

            out = scalar_for(array.array("d", [0.0, 1.0]), 3)
            self.assertEqual(len(out), 2)
            self.assertAlmostEqual(float(out[0]), 6.0)
            self.assertAlmostEqual(float(out[1]), 7.0)
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload

    def test_scalar_for_inlines_helper_call_into_step_src(self):
        original_call_jit = jit_mod.call_jit
        original_register_offload = jit_mod.register_offload
        original_step_loop = jit_mod.call_jit_step_loop_f64

        seen = {"step_src": None}

        try:
            def fake_call_jit(_func, _args, _kwargs):
                return _func(*_args)

            def fake_register_offload(_func, _strategy, _return_type, src, _arg_names):
                if src is not None and isinstance(_arg_names, list) and _arg_names == ["x", "i"]:
                    seen["step_src"] = src
                return None

            jit_mod.call_jit = fake_call_jit
            jit_mod.register_offload = fake_register_offload
            jit_mod.call_jit_step_loop_f64 = None

            def helper_calc(a, b):
                return (a * b) + (a - b)

            @jit_mod.offload(strategy="jit", return_type="float")
            def scalar_for(seed, n):
                x = seed
                for i in range(int(n)):
                    x += helper_calc(x * 0.0001, float(i) * 0.001)
                return x

            out = scalar_for(1.0, 3.0)
            self.assertIsInstance(out, float)
            self.assertIsNotNone(seen["step_src"])
            self.assertNotIn("helper_calc", str(seen["step_src"]))
        finally:
            jit_mod.call_jit = original_call_jit
            jit_mod.register_offload = original_register_offload
            jit_mod.call_jit_step_loop_f64 = original_step_loop


if __name__ == "__main__":
    unittest.main()
