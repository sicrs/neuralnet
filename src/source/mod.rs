pub trait DataSource<D>: Iterator {
    fn iter(&self) -> std::slice::Iter<D>;
}