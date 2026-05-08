//! Minimal protobuf wire-format decoder.
//!
//! Decodes raw protobuf bytes into [`RawField`] records without any schema.
//! Higher-level helpers (`get_varint`, `get_submsg`, …) provide typed access.

pub const VARINT: u8 = 0;
pub const FIXED64: u8 = 1;
pub const LEN_DELIM: u8 = 2;
pub const FIXED32: u8 = 5;

#[inline]
pub fn decode_varint(data: &[u8], mut pos: usize) -> (u64, usize) {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        let b = data[pos];
        pos += 1;
        result |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (result, pos)
}

/// Raw protobuf field: (field_id, wire_type, value_start, value_end_or_varint)
pub struct RawField {
    pub field_id: u32,
    pub wire_type: u8,
    /// For VARINT: value stored directly.
    /// For LEN_DELIM/FIXED*: unused.
    pub varint_val: u64,
    pub slice_start: usize,
    pub slice_end: usize,
}

pub fn decode_fields(data: &[u8], start: usize, end: usize) -> Vec<RawField> {
    let mut fields = Vec::with_capacity(8);
    let mut pos = start;
    while pos < end {
        let (tag, new_pos) = decode_varint(data, pos);
        pos = new_pos;
        let field_id = (tag >> 3) as u32;
        let wire_type = (tag & 0x07) as u8;
        match wire_type {
            VARINT => {
                let (val, new_pos) = decode_varint(data, pos);
                pos = new_pos;
                fields.push(RawField {
                    field_id,
                    wire_type,
                    varint_val: val,
                    slice_start: 0,
                    slice_end: 0,
                });
            }
            FIXED64 => {
                fields.push(RawField {
                    field_id,
                    wire_type,
                    varint_val: 0,
                    slice_start: pos,
                    slice_end: pos + 8,
                });
                pos += 8;
            }
            LEN_DELIM => {
                let (len, new_pos) = decode_varint(data, pos);
                pos = new_pos;
                let len = len as usize;
                fields.push(RawField {
                    field_id,
                    wire_type,
                    varint_val: 0,
                    slice_start: pos,
                    slice_end: pos + len,
                });
                pos += len;
            }
            FIXED32 => {
                fields.push(RawField {
                    field_id,
                    wire_type,
                    varint_val: 0,
                    slice_start: pos,
                    slice_end: pos + 4,
                });
                pos += 4;
            }
            _ => break,
        }
    }
    fields
}

// ── Typed field accessors ────────────────────────────────────────────────

#[inline]
pub fn get_varint(fields: &[RawField], id: u32) -> Option<u64> {
    fields
        .iter()
        .rev()
        .find(|f| f.field_id == id && f.wire_type == VARINT)
        .map(|f| f.varint_val)
}

#[inline]
pub fn get_i32(fields: &[RawField], id: u32) -> i32 {
    get_varint(fields, id).unwrap_or(0) as i32
}

#[inline]
pub fn get_submsg<'a>(fields: &[RawField], id: u32, data: &'a [u8]) -> Option<&'a [u8]> {
    fields
        .iter()
        .rev()
        .find(|f| f.field_id == id && f.wire_type == LEN_DELIM)
        .map(|f| &data[f.slice_start..f.slice_end])
}

pub fn get_repeated_submsg<'a>(fields: &[RawField], id: u32, data: &'a [u8]) -> Vec<&'a [u8]> {
    fields
        .iter()
        .filter(|f| f.field_id == id && f.wire_type == LEN_DELIM)
        .map(|f| &data[f.slice_start..f.slice_end])
        .collect()
}
