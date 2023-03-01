use rand::{rngs::ThreadRng, Rng};


pub struct SGenerator {
    v: u64,
    index: u32
}

impl SGenerator {
    pub fn new(rng: &mut ThreadRng) -> Self {
        Self { v: rng.gen(), index: 0 }
    }

    pub fn gen(&mut self, rng: &mut ThreadRng) -> i32 {
        let result = (((self.v >> self.index) & 1) << 1) as i32 - 1;
        self.index += 1;
        if self.index > 64 {
            *self = SGenerator::new(rng);
        }
        result
    }
}
