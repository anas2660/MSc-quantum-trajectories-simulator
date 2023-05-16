use std::ops::{IndexMut, Index, Range};



/// 2-Dimensional Vec
#[derive(Clone)]
pub struct MVec<T: Sized> {
    inner: Vec<T>,
    nrows: usize,
    ncols: usize
}

impl<T: Clone> MVec<T> {
    pub unsafe fn alloc_zeroed(nrows: usize, ncols: usize) -> Self {
        let mut inner = Vec::with_capacity(nrows*ncols);
        unsafe { inner.set_len(nrows*ncols) };

        // memset 0s
        inner.iter_mut().for_each(|m| *m = unsafe { std::mem::zeroed() } );

        Self { inner, nrows, ncols }
    }

    pub fn with_capacity(nrows: usize, ncols: usize) -> Self {
        Self {
            inner: Vec::with_capacity(nrows*ncols),
            nrows,
            ncols,
        }
    }

    pub fn push(&mut self, value: T) { self.inner.push(value) }

    pub fn extend_from_slice(&mut self, other: &[T]) { self.inner.extend_from_slice(other) }

    pub fn as_slice(&self) -> &[T] { self.inner.as_slice() }

    pub fn row_count(&self) -> usize { self.nrows }

    pub fn transpose(&mut self) {
        let mut copy = MVec::with_capacity(self.ncols, self.nrows);

        for c in 0..self.ncols {
            for r in 0..self.nrows {
                copy.push(self[r][c].clone());
            }
        }

        std::mem::swap(&mut copy, self);
    }
 }


impl<T> Index<usize> for MVec<T> {
    type Output = [T];

    fn index(&self, index: usize) -> &Self::Output {
        &self.inner[index*self.ncols..(index+1)*self.ncols]
    }
}

impl<T> IndexMut<usize> for MVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.inner[index*self.ncols..(index+1)*self.ncols]
    }
}


impl<T> Index<Range<usize>> for MVec<T> {
    type Output = [T];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        &self.inner[index.start*self.ncols..(index.end)*self.ncols]
    }
}

impl<T> IndexMut<Range<usize>> for MVec<T> {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        &mut self.inner[index.start*self.ncols..(index.end)*self.ncols]
    }
}
