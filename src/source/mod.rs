/// DataSource designates an Iterator as a data source
pub trait DataSource<D>: Iterator<Item = D> {
    fn len(&self) -> usize;
}
