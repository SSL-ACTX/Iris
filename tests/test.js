const { NodeRuntime } = require('../index.js');
const assert = require('assert');
const { promisify } = require('util');
const sleep = promisify(setTimeout);

const rt = new NodeRuntime();

console.log("--- Myrmidon Node.js Functional Tests ---");

async function test_push_actor() {
    console.log("[1/3] Testing Push Actor...");
    
    let received = [];
    let resolve_done;
    const done_promise = new Promise(r => resolve_done = r);

    // 1. Spawn a simple callback actor
    const pid = rt.spawn((msg) => {
        if (msg.data) {
            const str = msg.data.toString();
            received.push(str);
            if (received.length === 2) {
                resolve_done();
            }
        }
    }, 100);

    // 2. Send messages
    rt.send(pid, Buffer.from("hello"));
    rt.send(pid, Buffer.from("world"));

    // 3. Wait for processing
    await done_promise;
    
    assert.strictEqual(received[0], "hello");
    assert.strictEqual(received[1], "world");
    
    // Stop actor
    rt.stop(pid);
    console.log("✅ Push Actor Passed");
}

async function test_mailbox_actor() {
    console.log("[2/3] Testing Mailbox Actor (Async/Await)...");

    // 1. Define an async actor logic
    // Note: Node wrapper doesn't support return values from actors directly,
    // so we use a side-channel (external variable/promise) to verify.
    let result = null;
    let resolve_finished;
    const finished = new Promise(r => resolve_finished = r);

    const handler = async (mailbox) => {
        // Await the first message
        const msg1 = await mailbox.recv(); 
        const str1 = msg1.data.toString();

        // Await the second message with timeout
        const msg2 = await mailbox.recv(1.0);
        const str2 = msg2.data.toString();

        // Check timeout logic (expecting None/Null)
        const msg3 = await mailbox.recv(0.1);

        result = {
            m1: str1,
            m2: str2,
            m3: msg3
        };
        resolve_finished();
    };

    const pid = rt.spawnWithMailbox(handler, 100);

    // 2. Send messages
    rt.send(pid, Buffer.from("async"));
    rt.send(pid, Buffer.from("rocks"));
    // We do NOT send a 3rd message to test timeout

    // 3. Wait for actor to finish
    await finished;

    assert.strictEqual(result.m1, "async");
    assert.strictEqual(result.m2, "rocks");
    assert.strictEqual(result.m3, null, "Expected timeout to return null");

    rt.stop(pid);
    console.log("✅ Mailbox Actor Passed");
}

async function test_registry() {
    console.log("[3/3] Testing Name Registry...");

    let received = null;
    let resolve_done;
    const done = new Promise(r => resolve_done = r);

    // Spawn actor
    const pid = rt.spawn((msg) => {
        if (msg.data) {
            received = msg.data.toString();
            resolve_done();
        }
    });

    // Register name
    rt.register("my-service", pid);

    // Resolve locally
    const resolved_pid = rt.resolve("my-service");
    assert.strictEqual(resolved_pid, pid);

    // Send by name
    rt.sendNamed("my-service", Buffer.from("named_msg"));

    await done;
    assert.strictEqual(received, "named_msg");

    rt.stop(pid);
    console.log("✅ Registry Passed");
}

(async () => {
    try {
        await test_push_actor();
        await test_mailbox_actor();
        await test_registry();
        console.log("\n🎉 All Tests Passed!");
    } catch (e) {
        console.error("\n❌ Test Failed:", e);
        process.exit(1);
    }
})();