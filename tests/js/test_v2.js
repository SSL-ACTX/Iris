const { NodeRuntime } = require('../index.js');
const assert = require('assert');

const rt = new NodeRuntime();

async function test_metrics_and_health() {
    console.log("Testing Node Metrics and Health...");
    const pid = rt.spawn((msg) => {
        if (msg.request) {
            rt.send(msg.request.replyTo, Buffer.from("pong"));
        }
    });

    rt.setActorHealth(pid, "busy");
    const info = rt.actorInfo(pid);
    assert.strictEqual(info.health, "Busy");

    const metrics = rt.getMetrics();
    assert.ok(metrics.actorCount >= 1);

    console.log("Testing Node call()...");
    const res = await rt.call(pid, Buffer.from("ping"), 1.0);
    assert.strictEqual(res.toString(), "pong");

    rt.stop(pid);
    console.log("✅ Metrics and Health Passed");
}

async function test_load_shedding() {
    console.log("Testing Node Load Shedding...");
    rt.setSystemCapacity(1);
    rt.setLoadShedding(true);

    const pid1 = rt.spawn((m) => {});
    assert.ok(pid1 !== 0, "First spawn should work");

    const pid2 = rt.spawn((m) => {});
    // In JS, spawn returns i64, if it returned 0 on Rust side, it will be 0 here.
    assert.strictEqual(pid2, 0, "Second spawn should be rejected (0)");

    rt.stop(pid1);
    // wait a bit for telemetry to update
    await new Promise(r => setTimeout(r, 100));
    
    const pid3 = rt.spawn((m) => {});
    assert.ok(pid3 !== 0, "Spawn should work again after stop");
    rt.stop(pid3);

    console.log("✅ Load Shedding Passed");
}

(async () => {
    try {
        await test_metrics_and_health();
        await test_load_shedding();
        console.log("🎉 Node v2 Tests Passed!");
    } catch (e) {
        console.error("❌ Test Failed:", e);
        process.exit(1);
    }
})();
