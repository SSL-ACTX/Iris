// src/network.rs
//! Phase 5 & 7: Distributed Networking and Remote Resolution

use crate::mailbox::Message;
use crate::pid::Pid;
use bytes::{Bytes, BytesMut, Buf, BufMut};
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

pub struct NetworkManager {
    runtime: Arc<crate::Runtime>,
}

impl NetworkManager {
    pub fn new(runtime: Arc<crate::Runtime>) -> Self {
        Self { runtime }
    }

    pub async fn start_server(&self, addr: &str) -> std::io::Result<()> {
        let listener = TcpListener::bind(addr).await?;
        let rt = self.runtime.clone();

        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let rt_inner = rt.clone();
                tokio::spawn(async move {
                    let mut head = [0u8; 1];
                    while socket.read_exact(&mut head).await.is_ok() {
                        match head[0] {
                            0 => { // User Message: [PID:u64][LEN:u32][DATA]
                                let mut meta = [0u8; 12];
                                socket.read_exact(&mut meta).await.unwrap();
                                let mut cursor = std::io::Cursor::new(&meta);
                                let pid = cursor.get_u64();
                                let len = cursor.get_u32() as usize;
                                let mut data = vec![0u8; len];
                                socket.read_exact(&mut data).await.unwrap();
                                let _ = rt_inner.send(pid, Message::User(Bytes::from(data)));
                            }
                            1 => { // Resolve Request: [LEN:u32][NAME:String] -> [PID:u64]
                                let mut len_buf = [0u8; 4];
                                socket.read_exact(&mut len_buf).await.unwrap();
                                let len = u32::from_be_bytes(len_buf) as usize;
                                let mut name_vec = vec![0u8; len];
                                socket.read_exact(&mut name_vec).await.unwrap();
                                let name = String::from_utf8_lossy(&name_vec);

                                let pid = rt_inner.resolve(&name).unwrap_or(0);
                                socket.write_all(&pid.to_be_bytes()).await.unwrap();
                            }
                            _ => break,
                        }
                    }
                });
            }
        });
        Ok(())
    }

    pub async fn resolve_remote(&self, addr: &str, name: &str) -> std::io::Result<Pid> {
        let mut stream = TcpStream::connect(addr).await?;
        stream.write_all(&[1u8]).await?; // Type 1: Resolve
        let name_bytes = name.as_bytes();
        stream.write_all(&(name_bytes.len() as u32).to_be_bytes()).await?;
        stream.write_all(name_bytes).await?;

        let mut pid_buf = [0u8; 8];
        stream.read_exact(&mut pid_buf).await?;
        Ok(u64::from_be_bytes(pid_buf))
    }

    pub async fn send_remote(&self, addr: &str, pid: Pid, data: Bytes) -> std::io::Result<()> {
        let mut stream = TcpStream::connect(addr).await?;
        stream.write_all(&[0u8]).await?; // Type 0: Send
        let mut buf = BytesMut::with_capacity(12 + data.len());
        buf.put_u64(pid);
        buf.put_u32(data.len() as u32);
        buf.put(data);
        stream.write_all(&buf).await?;
        Ok(())
    }
}
