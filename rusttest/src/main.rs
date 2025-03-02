use rand::Rng;
use fastrand;
use rand_xorshift::XorShiftRng;
use std::time::Instant;
use std::iter::repeat_with;

const N: i64 = 68719476736;
const n: usize = 65536;
fn main() {
    let rngstart = Instant::now();
    let mut rng = fastrand::Rng::new();
    let mut dat: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            dat[i][j] = rng.f64();
        }
    }
    println!("Loop time taken: {:.2?}", rngstart.elapsed());
    let rngstart2 = Instant::now();
    let fast: Vec<Vec<f64>> = (0..n).map(|_| repeat_with(|| rng.f64()).take(n).collect()).collect();
    println!("Fastrand time taken: {:.2?}", rngstart2.elapsed());
    let a: f64 = dat.iter().map(|x| x.iter().sum::<f64>()).sum();
    let b: f64 = fast.iter().map(|x| x.iter().sum::<f64>()).sum();
}