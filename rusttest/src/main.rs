use rand::Rng;
use fastrand;
use rand_xorshift::XorShiftRng;
use std::time::Instant;

const N: i64 = 68719476736;

fn main() {
    let rngstart = Instant::now();
    let mut rng = rand::rng();
    let norm = (0..N).map(|_| rng.random_range(0.0_f64..1.0));
    println!("Normal Random Time taken: {:.2?}", rngstart.elapsed());
    let rngstart2 = Instant::now();
    let fast = (0..N).map(|_| fastrand::f64());
    println!("Fastrand time taken: {:.2?}", rngstart2.elapsed());
    let a: f64 = norm.sum();
    let b: f64 = fast.sum();
}