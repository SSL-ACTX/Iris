'use strict';

const { NodeRuntime } = require('./index.js');

(async function main() {

const rt = new NodeRuntime();

const MSG_COUNT = 100_000;
const MSG = Buffer.from("ping");
const latencies = new Array(MSG_COUNT);

let received = 0;
let startSend = 0;
let doneResolve;

/* ================= ACTOR ================= */

function hotActor(msg) {
    const now = process.hrtime.bigint();
    const sentAt = BigInt(msg.readBigInt64BE(0));
    const latencyNs = now - sentAt;

    latencies[received++] = Number(latencyNs) / 1e6; // ms

    if (received === MSG_COUNT) {
        doneResolve();
    }
}

/* ================= UTIL ================= */

function mb(n) {
    return (n / 1024 / 1024).toFixed(1);
}

function memStats(label = "") {
    const m = process.memoryUsage();
    console.log(
        `${label}🧠 Heap ${mb(m.heapUsed)}/${mb(m.heapTotal)} MB | RSS ${mb(m.rss)} MB`
    );
}

function percentile(sorted, p) {
    const idx = Math.floor(sorted.length * p);
    return sorted[Math.min(idx, sorted.length - 1)];
}

/* ================= HEADER ================= */

console.log(`\n--- Hot Actor Latency Test ---`);
console.log(`Messages: ${MSG_COUNT.toLocaleString()}\n`);

/* ================= SPAWN ================= */

const pid = rt.spawn(hotActor, 10);
console.log("Hot actor spawned");

/* ================= SEND ================= */

console.log("\nSending messages...");
startSend = process.hrtime.bigint();

const done = new Promise(r => doneResolve = r);

for (let i = 0; i < MSG_COUNT; i++) {
    const buf = Buffer.allocUnsafe(8);
    buf.writeBigInt64BE(process.hrtime.bigint(), 0);
    rt.send(pid, buf);
}

await done;

const end = process.hrtime.bigint();
const totalMs = Number(end - startSend) / 1e6;

console.log(`✅ Processed ${MSG_COUNT.toLocaleString()} messages`);
console.log(`⚡ Throughput: ${(MSG_COUNT / (totalMs / 1000)).toFixed(0)} msgs/sec`);

memStats("Before drain ");

/* ================= DRAIN ================= */

await new Promise(r => setImmediate(r));
await new Promise(r => setImmediate(r));

if (global.gc) {
    console.log("Running GC...");
    global.gc();
}

memStats("After drain  ");

/* ================= LATENCY STATS ================= */

latencies.sort((a, b) => a - b);

console.log("\nLatency (ms):");
console.log(`  p50  : ${percentile(latencies, 0.50).toFixed(3)}`);
console.log(`  p95  : ${percentile(latencies, 0.95).toFixed(3)}`);
console.log(`  p99  : ${percentile(latencies, 0.99).toFixed(3)}`);
console.log(`  max  : ${latencies[latencies.length - 1].toFixed(3)}`);

/* ================= CLEANUP ================= */

rt.stop(pid);

if (global.gc) {
    global.gc();
    memStats("After stop  ");
}

console.log("\n✔ Hot actor test complete.\n");

})();
