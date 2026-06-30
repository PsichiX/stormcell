/// A lightweight handle to a value stored in an [`Allocator`].
///
/// An address is a `(page, index)` pair, not a pointer: it stays valid as long
/// as the owning allocator is not [`pop`](Allocator::pop)ped past it, and can be
/// freely copied. The all-`u32::MAX` address is reserved as [`Address::INVALID`]
/// (the [`Default`]).
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Address {
    page: u32,
    index: u32,
}

impl Default for Address {
    #[inline]
    fn default() -> Self {
        Self::INVALID
    }
}

impl Address {
    /// The sentinel address that refers to no value.
    pub const INVALID: Self = Self {
        page: u32::MAX,
        index: u32::MAX,
    };

    /// Returns `true` unless this is [`Address::INVALID`].
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.page != u32::MAX && self.index != u32::MAX
    }

    /// The index of the page this address points into.
    #[inline]
    pub fn page(&self) -> u32 {
        self.page
    }

    /// The slot within the page this address points into.
    #[inline]
    pub fn index(&self) -> u32 {
        self.index
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "@{}#{}", self.page, self.index)
    }
}

/// A paged arena that owns `Copy` values and addresses them by [`Address`].
///
/// Storage grows one fixed-capacity page (`Vec<T>`) at a time, so existing
/// values are never moved and their addresses stay stable. The allocator is
/// stack-like: [`push`](Allocator::push) appends and [`pop`](Allocator::pop)
/// removes the most recently pushed value. There is no random deallocation -
/// the transformation pipeline rebuilds grids into a fresh allocator rather
/// than freeing individual cells.
#[derive(Clone)]
pub struct Allocator<T: Copy> {
    page_capacity: u32,
    pages: Vec<Vec<T>>,
}

impl<T: Copy> Allocator<T> {
    /// Creates an empty allocator whose pages each hold up to `page_capacity`
    /// values, pre-reserving room for `pages_capacity` pages.
    pub fn new(page_capacity: u32, pages_capacity: u32) -> Self {
        Self {
            page_capacity,
            pages: Vec::with_capacity(pages_capacity as usize),
        }
    }

    /// The number of allocated pages.
    pub fn pages_count(&self) -> usize {
        self.pages.len()
    }

    /// The number of values stored in the current (last) page.
    pub fn current_page_size(&self) -> usize {
        self.pages.last().map(|page| page.len()).unwrap_or_default()
    }

    /// Stores `data` and returns its [`Address`].
    ///
    /// # Panics
    /// Panics if the number of pages would exceed `u32::MAX`.
    pub fn push(&mut self, data: T) -> Address {
        if self.pages.len() == self.pages.capacity() {
            self.pages.reserve(self.pages.len());
        }
        if self.pages.is_empty()
            || self.pages.last().unwrap().len() == self.pages.last().unwrap().capacity()
        {
            if self.pages.len() >= u32::MAX as usize {
                panic!("CellAllocator: reached maximum number of pages");
            }
            self.pages
                .push(Vec::with_capacity(self.page_capacity as usize));
        }
        let page = self.pages.last_mut().unwrap();
        page.push(data);
        let index = page.len() as u32 - 1;
        Address {
            page: self.pages.len() as u32 - 1,
            index,
        }
    }

    /// Removes and returns the most recently pushed value, if any.
    ///
    /// Note that this invalidates the [`Address`] that referred to it; popping
    /// is only safe for addresses not still in use elsewhere (used during merge
    /// to discard children that collapsed into a single leaf).
    pub fn pop(&mut self) -> Option<T> {
        for page in self.pages.iter_mut().rev() {
            if let Some(data) = page.pop() {
                if page.is_empty() {
                    self.pages.pop();
                }
                return Some(data);
            }
        }
        None
    }

    /// Borrows the value at `address`, or `None` if the address is out of range.
    pub fn read(&self, address: Address) -> Option<&T> {
        self.pages
            .get(address.page as usize)
            .and_then(|page| page.get(address.index as usize))
    }

    /// Mutably borrows the value at `address`, or `None` if out of range.
    pub fn write(&mut self, address: Address) -> Option<&mut T> {
        self.pages
            .get_mut(address.page as usize)
            .and_then(|page| page.get_mut(address.index as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cell_allocator() {
        let mut allocator = Allocator::<f32>::new(3, 10);

        let items = std::array::from_fn::<Address, 13, _>(|index| allocator.push(index as f32));

        assert_eq!(allocator.pages_count(), 5);
        assert_eq!(allocator.current_page_size(), 1);
        for (index, address) in items.into_iter().enumerate() {
            assert_eq!(*allocator.read(address).unwrap(), index as f32);
        }
        assert_eq!(allocator.pop(), Some(12.0));
        assert_eq!(allocator.pages_count(), 4);
        assert_eq!(allocator.current_page_size(), 3);
    }
}
