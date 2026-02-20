const { NodeRuntime } = require('./index.js');

const rt = new NodeRuntime();

function logicA(msg) { }
function logicB(msg) { msg.length; }

const pid = rt.spawn(logicA, 100);

async function main() {
    async function swapBlitz() {
        for (let i = 0; i < 5000; i++) {
            rt.hotSwap(pid, i % 2 === 0 ? logicB : logicA);
        }
    }

    async function sendBlitz() {
        for (let i = 0; i < 20000; i++) {
            rt.send(pid, Buffer.from("test_payload"));
        }
    }

    await Promise.all([swapBlitz(), sendBlitz()]);
    rt.stop(pid);
    console.log("Done");
}

main();
