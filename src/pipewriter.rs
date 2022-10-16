use nalgebra::*;
use std::io::Write;

pub enum PipeWriter {
    Closed,
    Opened(std::fs::File),
}

impl PipeWriter {
    pub fn open(path: &str) -> Self {
        if std::fs::metadata(path).is_ok() {
            Self::Opened(
                std::fs::File::options()
                    .write(true)
                    .read(false)
                    .open(path)
                    .unwrap(),
            )
        } else {
            Self::Closed
        }
    }

    pub fn write_vec3(&mut self, vec: Vector3<f32>) {
        match self {
            PipeWriter::Opened(pipe) => {
                let buf_data = vec.data.as_slice().as_ptr() as *const u8;
                let buf = unsafe {
                    std::slice::from_raw_parts(buf_data, std::mem::size_of::<Vector3<f32>>())
                };
                pipe.write(buf).unwrap();
            }
            PipeWriter::Closed => (),
        }
    }

    pub fn write_u32(&mut self, number: u32) {
        match self {
            PipeWriter::Opened(pipe) => {
                pipe.write(&number.to_le_bytes()).unwrap();
            }
            PipeWriter::Closed => (),
        }
    }

    pub fn is_opened(&self) -> bool {
        match self {
            PipeWriter::Closed => false,
            PipeWriter::Opened(_) => true,
        }
    }
}
