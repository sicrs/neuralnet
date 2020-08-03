pub trait DataSource<O> {
    fn iter(&self) -> std::slice::Iter<O>;
}
