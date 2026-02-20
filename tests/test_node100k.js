'use strict';

const { NodeRuntime } = require('./index.js');
const readline = require('readline');

(async function main() {

const rt = new NodeRuntime();

const COUNT = 100_000;
const handlers = [];
const MSG = Buffer.from("ping");

function noOp(msg) {}

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

function bar(pct, width = 20) {
    const filled = Math.round(pct * width);
    return '█'.repeat(filled) + '░'.repeat(width - filled);
}

function renderProgress(label, current, total, start) {
    const elapsed = (Date.now() - start) / 1000;
    const rate = current / (elapsed || 1);
    const pct = current / total;
    const eta = (total - current) / (rate || 1);

    readline.clearLine(process.stdout, 0);
    readline.cursorTo(process.stdout, 0);
    process.stdout.write(
        `${label}: [${bar(pct)}] ${(pct * 100).toFixed(1)}% ` +
        `| ${rate.toFixed(0)}/s | ETA ${eta.toFixed(2)}s`
    );
}

/* ================= HEADER ================= */

console.log(`\n--- Node.js Myrmidon Stress Test ---`);
console.log(`Actors: ${COUNT.toLocaleString()}\n`);

/* ================= SPAWN ================= */

console.log("Spawning actors...");
let start = Date.now();

for (let i = 0; i < COUNT; i++) {
    handlers.push(rt.spawn(noOp, 10));

    if (i % 1000 === 0 || i === COUNT - 1) {
        renderProgress("Spawn", i + 1, COUNT, start);
    }
}

let spawnTime = (Date.now() - start) / 1000;
process.stdout.write("\n");

console.log(`✅ Spawned ${COUNT.toLocaleString()} actors in ${spawnTime.toFixed(2)}s`);
console.log(`⚡ Spawn rate: ${(COUNT / spawnTime).toFixed(0)} actors/sec`);
memStats();

/* ================= SEND ================= */

console.log("\nSending messages...");
start = Date.now();

for (let i = 0; i < handlers.length; i++) {
    rt.send(handlers[i], MSG);

    if (i % 1000 === 0 || i === COUNT - 1) {
        renderProgress("Send ", i + 1, COUNT, start);
    }
}

let sendTime = (Date.now() - start) / 1000;
process.stdout.write("\n");

console.log(`✅ Sent ${COUNT.toLocaleString()} messages in ${sendTime.toFixed(2)}s`);
console.log(`⚡ Send rate: ${(COUNT / sendTime).toFixed(0)} msgs/sec`);
memStats("Before drain ");

/* ================= DRAIN ================= */

console.log("\nDraining event loop...");
await new Promise(resolve => setImmediate(resolve));
await new Promise(resolve => setImmediate(resolve));

if (global.gc) {
    console.log("Running GC...");
    global.gc();
}

memStats("After drain  ");

/* ================= STOP ================= */

console.log("\nStopping actors...");
for (const pid of handlers) {
    rt.stop(pid);
}

if (global.gc) {
    global.gc();
    memStats("After stop  ");
}

console.log("\n✔ Test complete.\n");

})();
