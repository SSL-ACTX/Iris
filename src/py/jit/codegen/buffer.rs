// src/py/jit/codegen/buffer.rs

use crate::py::jit::codegen::BufferElemType;
use pyo3::prelude::*;
use std::ffi::CStr;

// helper for zero-copy buffer access used by the JIT runner
pub(crate) struct BufferView {
    pub(crate) view: pyo3::ffi::Py_buffer,
    pub(crate) elem_type: BufferElemType,
    pub(crate) len: usize,
}

impl BufferView {
    #[inline(always)]
    pub(crate) fn as_ptr_u8(&self) -> *const u8 {
        self.view.buf as *const u8
    }

    #[inline(always)]
    pub(crate) fn as_ptr_f64(&self) -> *const f64 {
        self.view.buf as *const f64
    }

    #[inline(always)]
    pub(crate) fn is_aligned_for_f64(&self) -> bool {
        (self.view.buf as usize).wrapping_rem(std::mem::align_of::<f64>()) == 0
    }
}

impl Drop for BufferView {
    fn drop(&mut self) {
        unsafe { pyo3::ffi::PyBuffer_Release(&mut self.view) };
    }
}

#[cfg(feature = "pyo3")]
pub(crate) unsafe fn parse_buffer_elem_type(view: &pyo3::ffi::Py_buffer) -> Option<BufferElemType> {
    if view.itemsize <= 0 {
        return None;
    }
    let itemsize = view.itemsize as usize;

    fn expected_size_for_code(code: char) -> Option<usize> {
        match code {
            'd' => Some(std::mem::size_of::<f64>()),
            'f' => Some(std::mem::size_of::<f32>()),
            'q' => Some(std::mem::size_of::<i64>()),
            'i' => Some(std::mem::size_of::<i32>()),
            'h' => Some(std::mem::size_of::<i16>()),
            'b' => Some(std::mem::size_of::<i8>()),
            'Q' => Some(std::mem::size_of::<u64>()),
            'I' => Some(std::mem::size_of::<u32>()),
            'H' => Some(std::mem::size_of::<u16>()),
            'B' => Some(std::mem::size_of::<u8>()),
            '?' => Some(std::mem::size_of::<u8>()),
            _ => None,
        }
    }

    fn to_elem_type(code: char, itemsize: usize) -> Option<BufferElemType> {
        if code == 'l' {
            return match itemsize {
                8 => Some(BufferElemType::I64),
                4 => Some(BufferElemType::I32),
                _ => None,
            };
        }
        if code == 'L' {
            return match itemsize {
                8 => Some(BufferElemType::U64),
                4 => Some(BufferElemType::U32),
                _ => None,
            };
        }
        if let Some(expected) = expected_size_for_code(code) {
            if expected != itemsize {
                return None;
            }
        }
        match code {
            'd' => Some(BufferElemType::F64),
            'f' => Some(BufferElemType::F32),
            'q' => Some(BufferElemType::I64),
            'i' => Some(BufferElemType::I32),
            'h' => Some(BufferElemType::I16),
            'b' => Some(BufferElemType::I8),
            'Q' => Some(BufferElemType::U64),
            'I' => Some(BufferElemType::U32),
            'H' => Some(BufferElemType::U16),
            'B' => Some(BufferElemType::U8),
            '?' => Some(BufferElemType::Bool),
            _ => None,
        }
    }

    if view.format.is_null() {
        return match itemsize {
            8 => Some(BufferElemType::F64),
            4 => Some(BufferElemType::F32),
            2 => Some(BufferElemType::I16),
            1 => Some(BufferElemType::U8),
            _ => None,
        };
    }

    let fmt = CStr::from_ptr(view.format).to_str().ok()?;
    let code = fmt
        .chars()
        .rev()
        .find(|ch| ch.is_ascii_alphabetic() || *ch == '?')?;
    to_elem_type(code, itemsize)
}

#[cfg(feature = "pyo3")]
pub(crate) unsafe fn open_typed_buffer(obj: &Bound<'_, PyAny>) -> Option<BufferView> {
    let mut view: pyo3::ffi::Py_buffer = std::mem::zeroed();
    let flags = pyo3::ffi::PyBUF_C_CONTIGUOUS | pyo3::ffi::PyBUF_FORMAT;
    if pyo3::ffi::PyObject_GetBuffer(obj.as_ptr(), &mut view, flags) != 0 {
        pyo3::ffi::PyErr_Clear();
        return None;
    }

    let itemsize = view.itemsize as usize;
    if itemsize == 0 {
        pyo3::ffi::PyBuffer_Release(&mut view);
        return None;
    }

    let elem_type = match parse_buffer_elem_type(&view) {
        Some(elem) => elem,
        None => {
            pyo3::ffi::PyBuffer_Release(&mut view);
            return None;
        }
    };

    let total_bytes = view.len as usize;
    if !total_bytes.is_multiple_of(itemsize) {
        pyo3::ffi::PyBuffer_Release(&mut view);
        return None;
    }

    let len = total_bytes / itemsize;
    Some(BufferView {
        view,
        elem_type,
        len,
    })
}

#[cfg(feature = "pyo3")]
#[inline(always)]
pub(crate) unsafe fn read_buffer_f64(view: &BufferView, index: usize) -> f64 {
    let base = view.as_ptr_u8();
    match view.elem_type {
        BufferElemType::F64 => {
            let p = base.add(index * std::mem::size_of::<f64>()) as *const f64;
            std::ptr::read_unaligned(p)
        }
        BufferElemType::F32 => {
            let p = base.add(index * std::mem::size_of::<f32>()) as *const f32;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::I64 => {
            let p = base.add(index * std::mem::size_of::<i64>()) as *const i64;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::I32 => {
            let p = base.add(index * std::mem::size_of::<i32>()) as *const i32;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::I16 => {
            let p = base.add(index * std::mem::size_of::<i16>()) as *const i16;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::I8 => {
            let p = base.add(index * std::mem::size_of::<i8>()) as *const i8;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::U64 => {
            let p = base.add(index * std::mem::size_of::<u64>()) as *const u64;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::U32 => {
            let p = base.add(index * std::mem::size_of::<u32>()) as *const u32;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::U16 => {
            let p = base.add(index * std::mem::size_of::<u16>()) as *const u16;
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::U8 => {
            let p = base.add(index * std::mem::size_of::<u8>());
            std::ptr::read_unaligned(p) as f64
        }
        BufferElemType::Bool => {
            let p = base.add(index);
            if std::ptr::read_unaligned(p) == 0 {
                0.0
            } else {
                1.0
            }
        }
    }
}
