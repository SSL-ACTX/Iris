//! Distributed Networking, Remote Resolution, and Heartbeat Monitoring

use crate::core::mailbox::{self, Message};
use crate::core::pid::Pid;
use bytes::{BufMut, Bytes, BytesMut};
use dashmap::DashMap;
use once_cell::sync::Lazy;
use std::io;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{timeout, Duration};
use tracing::{debug, info, warn};

use super::Runtime;

/// Current binary protocol version.  Peers must match or the connection is
/// immediately dropped.  Future releases will bump this when the wire format
/// changes; nodes on mixed-version clusters should handle rejections gracefully.
const PROTOCOL_VERSION: u8 = 1;

/// How large a single user payload can be before the connection is torn down.
#[allow(dead_code)]
const MAX_PAYLOAD_LEN: usize = 1024 * 1024; // 1 MiB

/// How many bytes a service name may contain when doing a remote resolve.
#[allow(dead_code)]
const MAX_NAME_LEN: usize = 1024;

/// Global connection pool for multiplexing outgoing messages to remote peers.
static PEER_POOL: Lazy<DashMap<String, tokio::sync::mpsc::Sender<Bytes>>> = Lazy::new(DashMap::new);

/// Types of messages that can appear on the wire.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum MessageType {
    User = 0,
    Resolve = 1,
    Ping = 2,
    PingPid = 3,
}

impl TryFrom<u8> for MessageType {
    type Error = ();
    fn try_from(b: u8) -> Result<Self, Self::Error> {
        match b {
            0 => Ok(MessageType::User),
            1 => Ok(MessageType::Resolve),
            2 => Ok(MessageType::Ping),
            3 => Ok(MessageType::PingPid),
            _ => Err(()),
        }
    }
}

pub struct NetworkManager {
    runtime: Arc<Runtime>,
}

impl NetworkManager {
    pub fn new(runtime: Arc<Runtime>) -> Self {
        Self { runtime }
    }

    /// Starts the TCP server for inter-node communication.
    pub async fn start_server(&self, addr: &str) -> io::Result<std::net::SocketAddr> {
        let listener = TcpListener::bind(addr).await?;
        let actual_addr = listener.local_addr()?;
        info!(%actual_addr, "network server listening");
        let rt = self.runtime.clone();

        tokio::spawn(async move {
            loop {
                match listener.accept().await {
                    Ok((socket, peer)) => {
                        debug!(%peer, "accepted connection");
                        let rt_inner = rt.clone();
                        tokio::spawn(async move {
                            if let Err(e) = Self::handle_connection(socket, rt_inner).await {
                                debug!(error = %e, "connection handler terminated");
                            }
                        });
                    }
                    Err(e) => {
                        warn!(error = %e, "error accepting connection, retrying");
                        continue;
                    }
                }
            }
        });

        Ok(actual_addr)
    }

    /// Helper that wraps a read operation with an adjustable timeout.
    async fn read_exact_with_timeout<R: AsyncReadExt + Unpin>(
        reader: &mut R,
        buf: &mut [u8],
        dur: Duration,
    ) -> io::Result<()> {
        let mut offset = 0;
        while offset < buf.len() {
            let n = match timeout(dur, reader.read(&mut buf[offset..])).await {
                Ok(Ok(0)) => {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "connection closed",
                    ))
                }
                Ok(Ok(n)) => n,
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err(io::Error::new(io::ErrorKind::TimedOut, "read timed out")),
            };
            offset += n;
        }
        Ok(())
    }

    /// Helper that wraps a write operation with an adjustable timeout.
    async fn write_all_with_timeout<W: AsyncWriteExt + Unpin>(
        writer: &mut W,
        buf: &[u8],
        dur: Duration,
    ) -> io::Result<()> {
        match timeout(dur, writer.write_all(buf)).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(io::Error::new(io::ErrorKind::TimedOut, "write timed out")),
        }
    }

    /// Core connection loop using buffered reads for performance.
    async fn handle_connection(socket: TcpStream, rt: Arc<Runtime>) -> io::Result<()> {
        let io_timeout = rt.get_network_io_timeout();
        let (read_half, mut write_half) = socket.into_split();
        let mut reader = BufReader::new(read_half);

        let mut version = [0u8; 1];
        Self::read_exact_with_timeout(&mut reader, &mut version, io_timeout).await?;
        if version[0] != PROTOCOL_VERSION {
            warn!(
                got = version[0],
                expected = PROTOCOL_VERSION,
                "protocol version mismatch"
            );
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "protocol version mismatch",
            ));
        }

        let mut header = [0u8; 1];
        while Self::read_exact_with_timeout(&mut reader, &mut header, io_timeout)
            .await
            .is_ok()
        {
            let msg_type = match MessageType::try_from(header[0]) {
                Ok(mt) => mt,
                Err(_) => {
                    warn!(byte = header[0], "unknown message type");
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "unknown message type",
                    ));
                }
            };

            match msg_type {
                MessageType::User => {
                    let mut meta = [0u8; 12];
                    Self::read_exact_with_timeout(&mut reader, &mut meta, io_timeout).await?;

                    let pid = u64::from_be_bytes(meta[0..8].try_into().unwrap());
                    let len = u32::from_be_bytes(meta[8..12].try_into().unwrap()) as usize;

                    if len > rt.get_network_max_payload() {
                        warn!(pid, len, "payload too large, dropping connection");
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "payload too large",
                        ));
                    }
                    let mut data = vec![0u8; len];
                    Self::read_exact_with_timeout(&mut reader, &mut data, io_timeout).await?;

                    let _ = rt.send(pid, Message::User(Bytes::from(data)));
                }
                MessageType::Resolve => {
                    let mut len_buf = [0u8; 4];
                    Self::read_exact_with_timeout(&mut reader, &mut len_buf, io_timeout).await?;

                    let len = u32::from_be_bytes(len_buf) as usize;
                    if len > rt.get_network_max_name_len() {
                        warn!(len, "resolve name too long");
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "resolve name too long",
                        ));
                    }

                    let mut name_vec = vec![0u8; len];
                    Self::read_exact_with_timeout(&mut reader, &mut name_vec, io_timeout).await?;

                    let name = match String::from_utf8(name_vec) {
                        Ok(s) => s,
                        Err(_) => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "invalid utf8 in resolve request",
                            ))
                        }
                    };

                    let pid = rt.resolve(&name).unwrap_or(0);
                    Self::write_all_with_timeout(&mut write_half, &pid.to_be_bytes(), io_timeout)
                        .await?;
                }
                MessageType::Ping => {
                    Self::write_all_with_timeout(&mut write_half, &[3u8], io_timeout).await?;
                }
                MessageType::PingPid => {
                    let mut pid_buf = [0u8; 8];
                    Self::read_exact_with_timeout(&mut reader, &mut pid_buf, io_timeout).await?;
                    let pid = u64::from_be_bytes(pid_buf);
                    let alive = rt.is_alive(pid);
                    let resp = if alive { 3u8 } else { 4u8 }; // 3=Pong, 4=Dead
                    Self::write_all_with_timeout(&mut write_half, &[resp], io_timeout).await?;
                }
            }
        }
        Ok(())
    }

    /// Queries a remote node for a PID associated with a name.
    pub async fn resolve_remote(&self, addr: &str, name: &str) -> io::Result<Pid> {
        debug!(%addr, %name, "resolving remote name");
        let io_timeout = self.runtime.get_network_io_timeout();
        let mut stream = timeout(io_timeout, TcpStream::connect(addr)).await??;

        Self::write_all_with_timeout(&mut stream, &[PROTOCOL_VERSION], io_timeout).await?;
        Self::write_all_with_timeout(&mut stream, &[MessageType::Resolve as u8], io_timeout)
            .await?;

        let name_bytes = name.as_bytes();
        if name_bytes.len() > self.runtime.get_network_max_name_len() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "name too long"));
        }

        Self::write_all_with_timeout(
            &mut stream,
            &(name_bytes.len() as u32).to_be_bytes(),
            io_timeout,
        )
        .await?;
        Self::write_all_with_timeout(&mut stream, name_bytes, io_timeout).await?;

        let mut pid_buf = [0u8; 8];
        Self::read_exact_with_timeout(&mut stream, &mut pid_buf, io_timeout).await?;
        Ok(u64::from_be_bytes(pid_buf))
    }

    /// Internal helper to fetch an existing connection to a peer, or provision a new one.
    async fn get_or_connect_peer(
        &self,
        addr: &str,
    ) -> io::Result<tokio::sync::mpsc::Sender<Bytes>> {
        if let Some(entry) = PEER_POOL.get(addr) {
            return Ok(entry.value().clone());
        }

        let io_timeout = self.runtime.get_network_io_timeout();
        let mut stream = timeout(io_timeout, TcpStream::connect(addr)).await??;
        Self::write_all_with_timeout(&mut stream, &[PROTOCOL_VERSION], io_timeout).await?;

        let (tx, mut rx) = tokio::sync::mpsc::channel::<Bytes>(1024);
        PEER_POOL.insert(addr.to_string(), tx.clone());

        let addr_clone = addr.to_string();
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if Self::write_all_with_timeout(&mut stream, &msg, io_timeout)
                    .await
                    .is_err()
                {
                    break;
                }
            }
            PEER_POOL.remove(&addr_clone);
        });

        Ok(tx)
    }

    /// Transmits a binary payload to a remote PID using a persistent multiplexed connection.
    pub async fn send_remote(&self, addr: &str, pid: Pid, data: Bytes) -> io::Result<()> {
        debug!(%addr, pid, "sending remote message");
        if data.len() > self.runtime.get_network_max_payload() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "payload too large",
            ));
        }

        let mut buf = BytesMut::with_capacity(13 + data.len());
        buf.put_u8(MessageType::User as u8);
        buf.put_u64(pid);
        buf.put_u32(data.len() as u32);
        buf.put_slice(&data);
        let msg = buf.freeze();

        let sender = self.get_or_connect_peer(addr).await?;
        if sender.send(msg.clone()).await.is_err() {
            // Pool connection died right as we tried to use it. Clear it and retry once.
            PEER_POOL.remove(addr);
            let sender2 = self.get_or_connect_peer(addr).await?;
            sender2.send(msg).await.map_err(|_| {
                io::Error::new(io::ErrorKind::ConnectionAborted, "remote node unavailable")
            })?;
        }

        Ok(())
    }

    /// Periodically probes a remote node to ensure it is alive.
    pub async fn monitor_remote(&self, addr: String, pid: Pid, interval_ms: u64) {
        let rt = self.runtime.clone();
        tokio::spawn(async move {
            let mut curr = Duration::from_millis(interval_ms);
            let base = curr;
            let factor = rt.get_monitor_backoff_factor();
            let max_backoff = rt.get_monitor_backoff_max();
            let threshold = rt.get_monitor_failure_threshold();
            let mut failures = 0;

            loop {
                // Prevent zombie monitors if the local PID was terminated
                if !rt.is_alive(pid) {
                    break;
                }

                let remote_pid = rt.remote_proxies.get(&pid).map(|e| e.value().1);

                tokio::time::sleep(curr).await;
                let io_timeout = rt.get_network_io_timeout();

                let is_up = match timeout(io_timeout, TcpStream::connect(&addr)).await {
                    Ok(Ok(mut stream)) => {
                        if Self::write_all_with_timeout(
                            &mut stream,
                            &[PROTOCOL_VERSION],
                            io_timeout,
                        )
                        .await
                        .is_ok()
                        {
                            if let Some(r_pid) = remote_pid {
                                // Specific actor monitoring
                                if Self::write_all_with_timeout(
                                    &mut stream,
                                    &[MessageType::PingPid as u8],
                                    io_timeout,
                                )
                                .await
                                .is_ok()
                                    && Self::write_all_with_timeout(
                                        &mut stream,
                                        &r_pid.to_be_bytes(),
                                        io_timeout,
                                    )
                                    .await
                                    .is_ok()
                                {
                                    let mut resp = [0u8; 1];
                                    Self::read_exact_with_timeout(
                                        &mut stream,
                                        &mut resp,
                                        io_timeout,
                                    )
                                    .await
                                    .is_ok()
                                        && resp[0] == 3 // 3=Pong (Alive)
                                } else {
                                    false
                                }
                            } else {
                                // Generic node monitoring
                                if Self::write_all_with_timeout(
                                    &mut stream,
                                    &[MessageType::Ping as u8],
                                    io_timeout,
                                )
                                .await
                                .is_ok()
                                {
                                    let mut pong = [0u8; 1];
                                    Self::read_exact_with_timeout(
                                        &mut stream,
                                        &mut pong,
                                        io_timeout,
                                    )
                                    .await
                                    .is_ok()
                                        && pong[0] == 3
                                } else {
                                    false
                                }
                            }
                        } else {
                            false
                        }
                    }
                    _ => false,
                };

                if !is_up {
                    failures += 1;
                    if failures >= threshold {
                        warn!(%addr, pid, "remote monitor detected node down after {} failures", failures);
                        rt.stop(pid);
                        rt.supervisor()
                            .notify_exit(pid, &mailbox::ExitReason::Disconnected);
                        break;
                    }
                    let next = Duration::from_secs_f64(curr.as_secs_f64() * factor);
                    curr = if next > max_backoff {
                        max_backoff
                    } else {
                        next
                    };
                } else {
                    failures = 0;
                    curr = base;
                }
            }
        });
    }
}

impl Runtime {
    /// Set the network I/O timeout used by all operations.
    pub fn set_network_io_timeout(&self, t: Duration) {
        *self.network_io_timeout.lock().unwrap() = t;
    }

    /// Get configured network I/O timeout.
    pub fn get_network_io_timeout(&self) -> Duration {
        *self.network_io_timeout.lock().unwrap()
    }

    /// Adjust maximum allowed payload length for send_remote (bytes).
    pub fn set_network_max_payload(&self, bytes: usize) {
        *self.network_max_payload.lock().unwrap() = bytes;
    }

    /// Get current payload limit.
    pub fn get_network_max_payload(&self) -> usize {
        *self.network_max_payload.lock().unwrap()
    }

    /// Adjust maximum allowed name length for remote resolve.
    pub fn set_network_max_name_len(&self, bytes: usize) {
        *self.network_max_name_len.lock().unwrap() = bytes;
    }

    /// Get current name length limit.
    pub fn get_network_max_name_len(&self) -> usize {
        *self.network_max_name_len.lock().unwrap()
    }

    /// Configure exponential backoff parameters for `monitor_remote`.
    ///
    /// `factor` is multiplied after each failure, capped by `max`.
    /// `failure_threshold` is how many consecutive failures must occur before
    /// the supervisor is notified.
    pub fn set_monitor_backoff(&self, factor: f64, max: Duration, failure_threshold: usize) {
        *self.monitor_backoff_factor.lock().unwrap() = factor;
        *self.monitor_backoff_max.lock().unwrap() = max;
        *self.monitor_failure_threshold.lock().unwrap() = failure_threshold;
    }

    pub fn get_monitor_backoff_factor(&self) -> f64 {
        *self.monitor_backoff_factor.lock().unwrap()
    }
    pub fn get_monitor_backoff_max(&self) -> Duration {
        *self.monitor_backoff_max.lock().unwrap()
    }
    pub fn get_monitor_failure_threshold(&self) -> usize {
        *self.monitor_failure_threshold.lock().unwrap()
    }

    // --- Distributed Networking ---

    /// Enable the node to receive remote messages on the specified TCP address.
    pub fn listen(&self, addr: String) {
        let rt_handle = Arc::new(self.clone());
        super::RUNTIME.spawn(async move {
            let manager = NetworkManager::new(rt_handle);
            match manager.start_server(&addr).await {
                Ok(actual) => tracing::info!(%actual, "node is now listening for remote messages"),
                Err(e) => eprintln!("[Iris] Network Server Error: {}", e),
            }
        });
    }

    /// Helper used internally when the runtime first learns about a remote
    /// actor.  We spawn a tiny local "proxy" actor whose job is to forward
    /// all user messages over the network to the real PID residing on the
    /// remote node.  Proxies behave exactly like ordinary actors from the
    /// caller's perspective (supervision, backpressure, `is_alive`, etc).
    pub(crate) fn lookup_or_create_proxy(&self, addr: &str, remote_pid: Pid) -> Pid {
        // if the caller accidentally passes a proxy PID that's already been
        // created for this address, just reuse it rather than nesting
        // proxies.
        if let Some(entry) = self.remote_proxies.get(&remote_pid) {
            let existing_addr = &entry.value().0;
            if existing_addr == addr {
                return remote_pid;
            }
        }

        // next, check the reverse index so we can return an existing proxy
        // without spawning a new actor.
        if let Some(entry) = self.proxy_by_remote.get(&(addr.to_string(), remote_pid)) {
            return *entry.value();
        }

        let addr_string = addr.to_string();
        let remote_copy = remote_pid;
        let rt_clone = self.clone();
        // clone early so the closure borrow doesn't consume it permanently
        let addr_string_clone = addr_string.clone();

        // spawn a handler that simply relays user messages
        let proxy_pid = self.spawn_actor(move |mut rx| {
            let rt_inner = rt_clone.clone();
            let addr_inner = addr_string_clone.clone();
            async move {
                while let Some(msg) = rx.recv().await {
                    if let mailbox::Message::User(bytes) = msg {
                        // bypass the public `send_remote` path to avoid
                        // creating a second proxy for the same target.  send
                        // directly through the network manager.
                        let manager = NetworkManager::new(Arc::new(rt_inner.clone()));
                        let _ = manager
                            .send_remote(&addr_inner, remote_copy, bytes.clone())
                            .await;
                    }
                }
            }
        });

        // record mapping in both directions
        self.remote_proxies
            .insert(proxy_pid, (addr_string.clone(), remote_pid));
        self.proxy_by_remote
            .insert((addr_string.clone(), remote_pid), proxy_pid);

        // automatically monitor the remote node; if it dies the proxy will
        // be stopped which in turn notifies any watchers of the exit.
        self.monitor_remote(addr_string.clone(), proxy_pid);

        proxy_pid
    }

    /// Resolve a name on a remote node.
    /// This is an async call that queries the remote node's registry.  The
    /// `Pid` returned is not the raw remote pid – it is a local proxy that
    /// forwards messages to the remote actor, allowing the caller to treat the
    /// result exactly like a normal local PID.
    pub async fn resolve_remote_async(&self, addr: String, name: String) -> Option<Pid> {
        let manager = NetworkManager::new(Arc::new(self.clone()));
        match manager.resolve_remote(&addr, &name).await {
            Ok(0) => None, // Node returned 0, meaning not found
            Ok(pid) => {
                let proxy = self.lookup_or_create_proxy(&addr, pid);
                Some(proxy)
            }
            Err(e) => {
                eprintln!("[Iris] Remote Resolve Error: {}", e);
                None
            }
        }
    }

    /// Send a binary payload to a PID on a specific remote node.
    pub fn send_remote(&self, addr: String, pid: Pid, data: bytes::Bytes) {
        // convert to a proxy so that callers don't need to manage raw remote
        // PIDs themselves.  This will spawn a proxy actor the first time we
        // talk to a particular remote target.
        let local = self.lookup_or_create_proxy(&addr, pid);
        // route using the normal send path so that overflow policies and
        // virtual activation work identically.
        let _ = self.send_user(local, data);
    }

    /// Remote Monitoring with Heartbeat support.
    /// Periodically probes the remote node (default 1s interval) to detect failures.
    pub fn monitor_remote(&self, addr: String, pid: Pid) {
        // if this pid is already a proxy for the same address we can use it
        // directly, otherwise create/look up a proxy for the raw remote PID.
        let local = if let Some(entry) = self.remote_proxies.get(&pid) {
            let existing_addr = &entry.value().0;
            if existing_addr == &addr {
                pid
            } else {
                self.lookup_or_create_proxy(&addr, pid)
            }
        } else {
            self.lookup_or_create_proxy(&addr, pid)
        };

        let rt_handle = Arc::new(self.clone());
        super::RUNTIME.spawn(async move {
            let manager = NetworkManager::new(rt_handle.clone());
            // Probes node health at a 1000ms interval for silent failure detection
            manager.monitor_remote(addr, local, 1000).await;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Runtime;
    use bytes::BytesMut;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::net::TcpStream;
    use tokio::time::Duration;

    async fn setup_runtime_and_server() -> (Arc<Runtime>, std::net::SocketAddr) {
        let rt = Arc::new(Runtime::new());
        let manager = NetworkManager::new(rt.clone());
        let addr = manager.start_server("127.0.0.1:0").await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        (rt, addr)
    }

    #[tokio::test]
    async fn oversized_payload_is_dropped_but_server_remains() {
        let (rt, addr) = setup_runtime_and_server().await;

        let counter = Arc::new(AtomicUsize::new(0));
        let c_clone = counter.clone();
        let pid = rt.spawn_handler_with_budget(
            move |msg| {
                let c_clone = c_clone.clone();
                async move {
                    if let Message::User(_buf) = msg {
                        c_clone.fetch_add(1, Ordering::SeqCst);
                    }
                }
            },
            10,
        );

        // send a valid message
        let mut stream = TcpStream::connect(&addr).await.unwrap();
        stream.write_all(&[PROTOCOL_VERSION]).await.unwrap();
        let data = b"hello";
        let mut buf = BytesMut::with_capacity(13 + data.len());
        buf.put_u8(MessageType::User as u8);
        buf.put_u64(pid);
        buf.put_u32(data.len() as u32);
        buf.put_slice(data);
        stream.write_all(&buf).await.unwrap();

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        assert!(rt.is_alive(pid));

        // send oversized message header only
        let mut stream2 = TcpStream::connect(&addr).await.unwrap();
        stream2.write_all(&[PROTOCOL_VERSION]).await.unwrap();
        let too_big = (MAX_PAYLOAD_LEN + 1) as u32;
        stream2.write_all(&[MessageType::User as u8]).await.unwrap();
        stream2.write_all(&pid.to_be_bytes()).await.unwrap();
        stream2.write_all(&too_big.to_be_bytes()).await.unwrap();

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // server still handles a later good message
        let mut stream3 = TcpStream::connect(&addr).await.unwrap();
        stream3.write_all(&[PROTOCOL_VERSION]).await.unwrap();
        let data2 = b"again";
        let mut buf2 = BytesMut::with_capacity(13 + data2.len());
        buf2.put_u8(MessageType::User as u8);
        buf2.put_u64(pid);
        buf2.put_u32(data2.len() as u32);
        buf2.put_slice(data2);
        stream3.write_all(&buf2).await.unwrap();

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn protocol_version_mismatch_closes_connection() {
        let (rt, addr) = setup_runtime_and_server().await;

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_spawn = counter.clone();
        let pid = rt.spawn_handler_with_budget(
            move |msg| {
                let c = counter_for_spawn.clone();
                async move {
                    if let Message::User(_buf) = msg {
                        c.fetch_add(1, Ordering::SeqCst);
                    }
                }
            },
            1,
        );

        let mut stream = TcpStream::connect(&addr).await.unwrap();
        stream.write_all(&[PROTOCOL_VERSION + 1]).await.unwrap();

        let data = b"will not be seen";
        let mut buf = BytesMut::with_capacity(13 + data.len());
        buf.put_u8(MessageType::User as u8);
        buf.put_u64(pid);
        buf.put_u32(data.len() as u32);
        buf.put_slice(data);
        let _ = stream.write_all(&buf).await;

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn configurable_limits_and_timeout() {
        let (rt, addr) = setup_runtime_and_server().await;
        rt.set_network_max_payload(5);
        rt.set_network_max_name_len(4);
        rt.set_network_io_timeout(Duration::from_millis(1));

        let manager = NetworkManager::new(rt.clone());
        let err = manager
            .send_remote(&addr.to_string(), 0, Bytes::from_static(b"longer"))
            .await
            .err();
        assert!(err.is_some());

        let res = manager.resolve_remote(&addr.to_string(), "toolong").await;
        assert!(res.is_err());

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let bad_addr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            if let Ok((_stream, _)) = listener.accept().await {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
        });

        let manager = NetworkManager::new(rt.clone());
        let e = manager
            .resolve_remote(&bad_addr.to_string(), "a")
            .await
            .unwrap_err();
        assert_eq!(e.kind(), std::io::ErrorKind::TimedOut);
    }

    #[tokio::test]
    async fn monitor_backoff_threshold_behavior() {
        let rt = Arc::new(Runtime::new());
        rt.set_monitor_backoff(2.0, Duration::from_millis(50), 2);

        let pid = rt.spawn_actor(|mut rx| async move {
            let _ = rx.recv().await;
        });
        let addr = "127.0.0.1:59999".to_string();
        let manager = NetworkManager::new(rt.clone());
        manager.monitor_remote(addr, pid, 10).await;

        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(rt.is_alive(pid));

        tokio::time::sleep(Duration::from_millis(60)).await;
        assert!(!rt.is_alive(pid));
    }

    #[tokio::test]
    async fn resolve_and_send_via_proxy() {
        // provider runtime holds the actual actor
        let (rt_a, addr) = setup_runtime_and_server().await;
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let svc = rt_a.spawn_handler_with_budget(
            move |msg| {
                let c = c.clone();
                async move {
                    if let Message::User(_b) = msg {
                        c.fetch_add(1, Ordering::SeqCst);
                    }
                }
            },
            10,
        );
        rt_a.register("svc".to_string(), svc);

        // client runtime resolves and then uses send() normally
        let rt_b = Arc::new(Runtime::new());
        let proxy = rt_b
            .resolve_remote_async(addr.to_string(), "svc".to_string())
            .await
            .expect("resolved should return pid");
        // numeric PID equality across separate runtimes is unimportant; the
        // real validation is that messages traverse the network below.

        rt_b.send(proxy, Message::User(Bytes::from_static(b"yo")))
            .unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
