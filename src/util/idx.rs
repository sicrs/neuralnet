use crate::{source::DataSource, Vector};
use std::{fs::File, io::Read};

struct MNISTLabelReader {
    num_items: usize,
    dimens: u8,
    inner: File,
}

#[inline]
fn u8_to_u32(bytes: &[u8]) -> u32 {
    let mut res: u32 = 0;
    for i in 0..4 {
        res |= (bytes[i] as u32) << (8 * (3 - i));
    }

    res
}

impl MNISTLabelReader {
    fn new(filename: &'static str) -> MNISTLabelReader {
        let mut inner: File = File::open(filename).unwrap();
        let mut buf: Vec<u8> = Vec::with_capacity(8);
        inner.read(&mut buf[0..8]).unwrap();
        /*
        let data_type: usize = match buf[2] {
            0x08 | 0x09 => 1,
            0x0B => 2,
            0x0C | 0x0D => 4,
            0x0E => 8,
            _ => panic!("Unrecognised data type!")
        };*/
        let dimens: u8 = buf[3];
        let num_items: u32 = u8_to_u32(&buf[4..8]);

        MNISTLabelReader {
            num_items: num_items as usize,
            dimens,
            inner,
        }
    }

    fn read_next_label(&mut self) -> Option<Vector> {
        let mut buf: Vec<u8> = Vec::with_capacity(1);
        let len = self.inner.read(&mut buf[0..1]).unwrap();
        if len == 0 {
            None
        } else {
            let mut vec_in = [0.0].repeat(10);
            vec_in[((buf[0] as usize) - 1)] = 1.0;
            Some(Vector::from(vec_in))
        }
    }
}

struct MNISTImageReader {
    inner: File,
    num_items: usize,
    n_rows: usize,
    n_cols: usize,
    n_pixels: usize,
}

impl MNISTImageReader {
    fn new(filename: &'static str) -> MNISTImageReader {
        let mut inner: File = File::open(filename).unwrap();
        let mut buf: Vec<u8> = Vec::with_capacity(16);
        inner.read(&mut buf[0..16]).unwrap();
        /*
        let data_type: usize = match buf[2] {
            0x08 | 0x09 => 1,
            0x0B => 2,
            0x0C | 0x0D => 4,
            0x0E => 8,
            _ => panic!("Unrecognised data type!")
        };*/
        let num_items: u32 = u8_to_u32(&buf[4..8]);
        let num_rows: u32 = u8_to_u32(&buf[8..12]);
        let num_cols: u32 = u8_to_u32(&buf[12..16]);
        
        MNISTImageReader {
            inner,
            num_items: num_items as usize,
            n_rows: num_rows as usize,
            n_cols: num_cols as usize,
            n_pixels: (num_cols * num_rows) as usize,
        }
    }

    fn read_next_image(&mut self) -> Option<Vector> {
        let mut buf: Vec<u8> = Vec::with_capacity(self.n_pixels);
        let len = self.inner.read(&mut buf[0..self.n_pixels]).unwrap();
        if len == 0 {
            None
        } else {
            let float_vec: Vec<f64> = buf[0..self.n_pixels].iter()
                .map(|x| *x as f64)
                .collect();
            Some(Vector::from(float_vec))
        }
    }
}

pub struct MNISTIDXTrainingData {
    images: MNISTImageReader,
    labels: MNISTLabelReader,
}

impl Iterator for MNISTIDXTrainingData {
    type Item = (Vector, Vector);
    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}
