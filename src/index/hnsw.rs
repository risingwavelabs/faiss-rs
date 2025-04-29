//! Interface and implementation to HNSWFlat index type.

//! Interface and implementation to IVFFlat index type.

use std::os::raw::c_int;
use std::slice;

use super::*;

pub struct Hnsw<'a> {
    inner: &'a *mut FaissIndexHNSW,
    levels_raw: &'a [c_int],
}

impl<'a> Hnsw<'a> {
    pub fn new(inner: &'a *mut FaissIndexHNSW) -> Self {
        let levels_raw = unsafe {
            let ntotal = faiss_Index_ntotal(*inner) as usize;
            let mut level_ptr: *const c_int = ptr::null();
            faiss_IndexHNSW_levels(*inner, &mut level_ptr);
            slice::from_raw_parts(level_ptr, ntotal)
        };
        Hnsw { inner, levels_raw }
    }
    pub fn entry_point(&self) -> Option<(usize, usize)> {
        let mut entry_point: idx_t = 0;
        let mut max_level: c_int = 0;
        unsafe {
            faiss_IndexHNSW_entry_point(*self.inner, &mut entry_point, &mut max_level);
            (entry_point >= 0).then_some((entry_point as _, max_level as _))
        }
    }

    pub fn levels_raw(&self) -> &'a [c_int] {
        self.levels_raw
    }

    pub fn neighbors_raw(&self, idx: usize, level: usize) -> &'a [FaissHNSWNeighborIdx] {
        assert!(idx < self.levels_raw.len());
        assert!(level < self.levels_raw[idx] as usize);
        unsafe {
            let mut neighbors_ptr = ptr::null();
            let mut neighbor_count = 0;
            faiss_IndexHNSW_neighbors(
                *self.inner,
                idx as _,
                level as _,
                &mut neighbors_ptr,
                &mut neighbor_count,
            );
            slice::from_raw_parts(neighbors_ptr, neighbor_count)
        }
    }
}

/// Alias for the native implementation of a HNSWFlat index.
pub type HnswFlatIndex = HnswFlatIndexImpl;

/// Native implementation of a flat index.
#[derive(Debug)]
pub struct HnswFlatIndexImpl {
    inner: *mut FaissIndexHNSWFlat,
}

unsafe impl Send for HnswFlatIndexImpl {}
unsafe impl Sync for HnswFlatIndexImpl {}

impl CpuIndex for HnswFlatIndexImpl {}

impl Drop for HnswFlatIndexImpl {
    fn drop(&mut self) {
        unsafe {
            faiss_IndexHNSWFlat_free(self.inner);
        }
    }
}

impl HnswFlatIndexImpl {
    fn new_helper(d: u32, m: u32, metric: MetricType) -> Result<Self> {
        unsafe {
            let metric = metric as c_uint;
            let mut inner = ptr::null_mut();
            faiss_try(faiss_IndexHNSWFlat_new_with_metric(
                &mut inner, d as c_int, m as c_int, metric,
            ))?;
            Ok(HnswFlatIndexImpl { inner })
        }
    }

    /// Create a new HNSW flat index.
    // The index owns the quantizer.
    pub fn new(d: u32, m: u32, metric: MetricType) -> Result<Self> {
        let index = HnswFlatIndexImpl::new_helper(d, m, metric)?;
        Ok(index)
    }

    /// Create a new HNSW flat index with L2 as the metric type.
    pub fn new_l2(d: u32, m: u32) -> Result<Self> {
        HnswFlatIndexImpl::new(d, m, MetricType::L2)
    }

    /// Create a new HNSW flat index with IP (inner product) as the metric type.
    pub fn new_ip(d: u32, m: u32) -> Result<Self> {
        HnswFlatIndexImpl::new(d, m, MetricType::InnerProduct)
    }

    pub fn hnsw(&self) -> Hnsw<'_> {
        Hnsw::new(&self.inner)
    }

    pub fn set_ef_construction(&mut self, ef_construction: usize) {
        unsafe {
            faiss_IndexHNSW_set_ef_construction(self.inner, ef_construction as c_int);
        }
    }

    pub fn set_ef_search(&mut self, ef_search: usize) {
        unsafe {
            faiss_IndexHNSW_set_ef_search(self.inner, ef_search as c_int);
        }
    }
}

impl NativeIndex for HnswFlatIndexImpl {
    fn inner_ptr(&self) -> *mut FaissIndex {
        self.inner
    }
}

impl FromInnerPtr for HnswFlatIndexImpl {
    unsafe fn from_inner_ptr(inner_ptr: *mut FaissIndex) -> Self {
        HnswFlatIndexImpl {
            inner: inner_ptr as *mut FaissIndexHNSWFlat,
        }
    }
}

impl_native_index!(HnswFlatIndex);

impl TryClone for HnswFlatIndexImpl {
    fn try_clone(&self) -> Result<Self>
    where
        Self: Sized,
    {
        try_clone_from_inner_ptr(self)
    }
}

impl_concurrent_index!(HnswFlatIndexImpl);

impl IndexImpl {
    /// Attempt a dynamic cast of an index to the HNSW flat index type.
    pub fn into_hnsw_flat(self) -> Result<HnswFlatIndexImpl> {
        unsafe {
            let new_inner = faiss_IndexHNSWFlat_cast(self.inner_ptr());
            if new_inner.is_null() {
                Err(Error::BadCast)
            } else {
                mem::forget(self);
                Ok(HnswFlatIndexImpl { inner: new_inner })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HnswFlatIndexImpl;
    use crate::index::{index_factory, ConcurrentIndex, Idx, Index, UpcastIndex};
    use crate::MetricType;

    const D: u32 = 8;

    #[test]
    // #[ignore]
    fn index_search() {
        let mut index = HnswFlatIndexImpl::new_l2(D, 5).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 4.,
            -4., -8., 1., 1., 2., 4., -1., 8., 8., 10., -10., -10., 10., -10., 10., 16., 16., 32.,
            25., 20., 20., 40., 15.,
        ];
        index.train(some_data).unwrap();
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let my_query = [0.; D as usize];
        let result = index.search(&my_query, 3).unwrap();
        assert_eq!(result.labels.len(), 3);
        assert!(result.labels.into_iter().all(Idx::is_some));
        assert_eq!(result.distances.len(), 3);
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = [100.; D as usize];
        // flat index can be used behind an immutable ref
        let result = (&index).search(&my_query, 3).unwrap();
        assert_eq!(result.labels.len(), 3);
        assert!(result.labels.into_iter().all(Idx::is_some));
        assert_eq!(result.distances.len(), 3);
        assert!(result.distances.iter().all(|x| *x > 0.));

        index.reset().unwrap();
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn index_search_own() {
        let mut index = HnswFlatIndexImpl::new_l2(D, 5).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 4.,
            -4., -8., 1., 1., 2., 4., -1., 8., 8., 10., -10., -10., 10., -10., 10., 16., 16., 32.,
            25., 20., 20., 40., 15.,
        ];
        index.train(some_data).unwrap();
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let my_query = [0.; D as usize];
        let result = index.search(&my_query, 3).unwrap();
        assert_eq!(result.labels.len(), 3);
        assert!(result.labels.into_iter().all(Idx::is_some));
        assert_eq!(result.distances.len(), 3);
        assert!(result.distances.iter().all(|x| *x > 0.));

        let my_query = [100.; D as usize];
        // flat index can be used behind an immutable ref
        let result = (&index).search(&my_query, 3).unwrap();
        assert_eq!(result.labels.len(), 3);
        assert!(result.labels.into_iter().all(Idx::is_some));
        assert_eq!(result.distances.len(), 3);
        assert!(result.distances.iter().all(|x| *x > 0.));

        index.reset().unwrap();
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn index_assign() {
        let mut index = HnswFlatIndexImpl::new_l2(D, 5).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 4.,
            -4., -8., 1., 1., 2., 4., -1., 8., 8., 10., -10., -10., 10., -10., 10., 16., 16., 32.,
            25., 20., 20., 40., 15.,
        ];
        index.train(some_data).unwrap();
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let my_query = [0.; D as usize];
        let result = index.assign(&my_query, 3).unwrap();
        assert_eq!(result.labels.len(), 3);
        assert!(result.labels.into_iter().all(Idx::is_some));

        let my_query = [100.; D as usize];
        // flat index can be used behind an immutable ref
        let result = (&index).assign(&my_query, 3).unwrap();
        assert_eq!(result.labels.len(), 3);
        assert!(result.labels.into_iter().all(Idx::is_some));

        index.reset().unwrap();
        assert_eq!(index.ntotal(), 0);
    }

    #[test]
    fn hnsw_flat_index_from_cast() {
        let mut index = index_factory(8, "HNSW5,Flat", MetricType::L2).unwrap();
        let some_data = &[
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];
        index.train(some_data).unwrap();
        index.add(some_data).unwrap();
        assert_eq!(index.ntotal(), 5);

        let index: HnswFlatIndexImpl = index.into_hnsw_flat().unwrap();
        assert_eq!(index.is_trained(), true);
        assert_eq!(index.ntotal(), 5);
    }

    #[test]
    fn index_upcast() {
        let index = HnswFlatIndexImpl::new_l2(D, 5).unwrap();
        assert_eq!(index.d(), D);
        assert_eq!(index.ntotal(), 0);

        let index_impl = index.upcast();
        assert_eq!(index_impl.d(), D);
    }
}
