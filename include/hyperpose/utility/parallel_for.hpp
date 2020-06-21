#pragma once

#if defined(HYPERPOSE_USE_CPP17_PARALLEL_FOR)
#include <type_traits>
#include <iterator>
#include <algorithm>
#include <execution>
#elif defined(HYPERPOSE_USE_PPL_PARALLEL_FOR)
#include <ppl.h>
#elif defined (HYPERPOSE_USE_TBB_PARALLEL_FOR)
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
//#include "tbb/blocked_range2d.h"
#else
#include <ttl/range>
#endif

namespace hyperpose
{
#if defined(HYPERPOSE_USE_CPP17_PARALLEL_FOR)
    namespace impl
    {
        // ttl::range's iterators are not random access iterator nor provide the required
        // nested types to be compatible with parallel std::for_each.
        template <typename IndexType>
        class integer_range final
        {
            static_assert(std::is_integral_v<IndexType>);
            const IndexType from_;
            const IndexType to_;

            class iterator final
            {
                IndexType pos_;
            public:
                using iterator_category = std::random_access_iterator_tag;
                using value_type = const IndexType;
                using difference_type = const value_type;
                using pointer = const value_type*;
                using reference = const value_type;

                constexpr inline iterator(const IndexType pos = IndexType(0)) : pos_(pos) {}

                constexpr inline iterator(const iterator&) = default;
                constexpr inline iterator(iterator&&) = default;

                constexpr inline iterator& operator=(const iterator&) = default;
                constexpr inline iterator& operator=(iterator&&) = default;

                constexpr inline iterator& operator+=(difference_type rhs) { pos_ += rhs; return *this; }
                constexpr inline iterator& operator-=(difference_type rhs) { pos_ -= rhs; return *this; }
                constexpr inline reference operator*() const { return pos_; }
                //constexpr inline Type* operator->() const { return &pos_; }
                //constexpr inline Type& operator[](difference_type rhs) const { ; }

                constexpr inline iterator& operator++() { ++pos_; return *this; }
                constexpr inline iterator& operator--() { --pos_; return *this; }
                constexpr inline iterator operator++(const int) const { iterator tmp(*this); ++pos_; return tmp; }
                constexpr inline iterator operator--(const int) const { iterator tmp(*this); --pos_; return tmp; }
                //constexpr inline iterator operator+(const iterator& rhs) {return iterator(_ptr+rhs.ptr);}
                constexpr inline difference_type operator-(const iterator& rhs) const { return (pos_ - rhs.pos_); }
                constexpr inline iterator operator+(difference_type rhs) const { return iterator(pos_ + rhs); }
                constexpr inline iterator operator-(difference_type rhs) const { return iterator(pos_ - rhs); }
                constexpr friend inline iterator operator+(difference_type lhs, const iterator& rhs) { return iterator(lhs + rhs.pos_); }
                constexpr friend inline iterator operator-(difference_type lhs, const iterator& rhs) { return iterator(lhs - rhs.pos_); }

                constexpr inline bool operator==(const iterator& rhs) const { return pos_ == rhs.pos_; }
                constexpr inline bool operator!=(const iterator& rhs) const { return pos_ != rhs.pos_; }
                constexpr inline bool operator>(const iterator& rhs) const { return pos_ > rhs.pos_; }
                constexpr inline bool operator<(const iterator& rhs) const { return pos_ < rhs.pos_; }
                constexpr inline bool operator>=(const iterator& rhs) const { return pos_ >= rhs.pos_; }
                constexpr inline bool operator<=(const iterator& rhs) const { return pos_ <= rhs.pos_; }
            };
        public:
            constexpr inline explicit integer_range(const IndexType m, const IndexType n): from_(m), to_(n) {}
            constexpr inline explicit integer_range(const IndexType n): integer_range(0, n) {}

            constexpr inline iterator begin() const { return iterator(from_); }
            constexpr inline iterator end() const { return iterator(to_); }
        };

        template <typename N>
        constexpr inline integer_range<N> range(const N n) { return integer_range<N>(n); }
        template <typename N>
        constexpr inline integer_range<N> range(const N m, const N n) { return integer_range<N>(m, n); }
    }
#endif

    template < typename IndexType, typename Function >
    inline void parallel_for(const IndexType range, Function&& fn)
    {
#if defined(HYPERPOSE_USE_CPP17_PARALLEL_FOR)

          const auto irange = impl::range<IndexType>(range);
          std::for_each(std::execution::par_unseq, irange.begin(), irange.end(), std::forward<Function>(fn));

#elif defined(HYPERPOSE_USE_PPL_PARALLEL_FOR)

        concurrency::parallel_for(std::size_t(0), static_cast<std::size_t>(range), std::forward<Function>(fn));

#elif defined (HYPERPOSE_USE_TBB_PARALLEL_FOR)

        tbb::parallel_for(tbb::blocked_range<IndexType>(IndexType(0), range), [=](const auto& brange) -> void
        {
            for (auto idx = brange.begin(); idx < brange.end(); ++idx)
                fn(idx);
        });

#else
        for (const auto k : ttl::range(range))
        {
            fn(k);
        }
#endif
    }

}