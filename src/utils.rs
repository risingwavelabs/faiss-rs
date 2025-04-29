/// L2-renormalize a set of vector. Nothing done if the vector is 0-normed
pub fn fvec_renorm_l2(d: usize, nx: usize, fvec: &mut [f32]) {
    unsafe { faiss_sys::faiss_fvec_renorm_L2(d, nx, fvec.as_mut_ptr()) }
}

pub fn fvec_inner_product(x: &[f32], y: &[f32]) -> f32 {
    let len = x.len();
    assert_eq!(len, y.len());
    let mut ret = 0.0;
    unsafe {
        faiss_sys::faiss_fvec_inner_products_ny(&mut ret, x.as_ptr(), y.as_ptr(), len, 1);
    }
    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: u32 = 8;

    #[test]
    fn check_fvec_renorm_l2_01() {
        let mut some_data = vec![
            7.5_f32, -7.5, 7.5, -7.5, 7.5, 7.5, 7.5, 7.5, -1., 1., 1., 1., 1., 1., 1., -1., 0., 0.,
            0., 1., 1., 0., 0., -1., 100., 100., 100., 100., -100., 100., 100., 100., 120., 100.,
            100., 105., -100., 100., 100., 105.,
        ];

        fvec_renorm_l2(D as usize, 5, &mut some_data);
    }

    #[test]
    fn check_fvec_inner_product() {
        let x = [1.1, 2.2];
        let y = [-2.9, 0.0];
        assert_eq!(fvec_inner_product(&x, &y), x[0] * y[0] + x[1] * y[1]);
    }
}
