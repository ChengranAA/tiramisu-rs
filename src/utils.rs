use ndarray::{ArrayViewD, Axis};

pub fn squeeze<A>(view: ArrayViewD<'_, A>) -> ArrayViewD<'_, A> {
    let mut v = view;
    let mut axis = 0;

    while axis < v.ndim() {
        if v.shape()[axis] == 1 {
            v = v.remove_axis(Axis(axis));
        } else {
            axis += 1;
        }
    }
    v
}
