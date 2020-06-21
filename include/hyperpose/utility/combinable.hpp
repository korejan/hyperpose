#pragma once

#if defined(HYPERPOSE_USE_TBB_PARALLEL_FOR)
    #include "tbb/combinable.h"
#elif defined(HYPERPOSE_USE_PPL_PARALLEL_FOR)
    #include <ppl.h>
#elif defined (HYPERPOSE_USE_CPP17_PARALLEL_FOR)
    #include <cassert>
    #include <functional>
    #include <algorithm>
    #include <unordered_map>
    #include <shared_mutex>
    #include <thread>
#else
    #include <functional>
#endif

namespace hyperpose
{
#if defined(HYPERPOSE_USE_TBB_PARALLEL_FOR)
	template < typename Tp >
	using combinable = tbb::combinable<Tp>;
#elif defined(HYPERPOSE_USE_PPL_PARALLEL_FOR)
	template < typename Tp >
	using combinable = concurrency::combinable<Tp>;
#elif defined (HYPERPOSE_USE_CPP17_PARALLEL_FOR)
    template < typename Tp >
    class combinable final
    {
        using tls_map_type = std::unordered_map<std::thread::id, Tp>;
        using init_fn_type = std::function< Tp() >;
        using mutext_type = std::shared_mutex;

        init_fn_type init_fn;
        tls_map_type tls_map;
        mutable mutext_type rw_map_mutex;

        inline tls_map_type clone_map() const
        {
            std::shared_lock<mutext_type> reader_lock(rw_map_mutex);
            return tls_map;
        }

    public:

        template <typename Function>
        inline explicit combinable(Function&& init_fun)
        : init_fn(std::forward<Function>(init_fun))
        {
            tls_map.reserve(std::thread::hardware_concurrency());
        }

        inline combinable()
        : combinable([]() { return Tp(); }) {}

        inline combinable(const combinable& c)
        : tls_map(clone_map()), init_fn(c.init_fn) {}

        inline combinable(combinable&& c)
        : init_fn(std::move(c.init_fn))
        {
            std::lock_guard<mutext_type> writer_lock(c.rw_map_mutex);
            tls_map = std::move(c.tls_map);
        }

        inline combinable& operator=(const combinable& c)
        {
            init_fn = c.init_fn;
            auto lmap = c.clone_map();
            {
                std::lock_guard<mutext_type> writer_lock(rw_map_mutex);
                tls_map = std::move(lmap);
            }
            return *this;
        }

        inline combinable& operator=(combinable&& c)
        {
            init_fn = std::move(c.init_fn);
            tls_map_type lmap;
            {
                std::lock_guard<mutext_type> writer_lock(c.rw_map_mutex);
                lmap = std::move(c.tls_map);
            }
            {
                std::lock_guard<mutext_type> writer_lock(rw_map_mutex);
                tls_map = std::move(lmap);
            }
            return *this;
        }

        inline void clear()
        {
            std::lock_guard<mutext_type> writer_lock(rw_map_mutex);
            tls_map.clear();
        }

        template < typename BinaryFunction >
        inline Tp combine(BinaryFunction&& binFn) const
        {
            std::shared_lock<mutext_type> reader_lock(rw_map_mutex);
            if (tls_map.size() < 2)
                return tls_map.empty() ? init_fn() : tls_map.begin()->second;

            auto first = tls_map.begin();
            const Tp& init = first->second;
            return std::reduce(++first, tls_map.end(), init, [&binFn](const auto& lhs, const auto& rhs)
            {
                return binFn(lhs, rhs.second);
            });
        }

        template < typename UnaryFunction >
        void combine_each(UnaryFunction&& fn) const
        {
            std::shared_lock<mutext_type> reader_lock(rw_map_mutex);
            for (const auto& [_, v] : tls_map)
                fn(v);
        }

        template < typename UnaryFunction >
        void combine_each(UnaryFunction&& fn)
        {
            std::lock_guard<mutext_type> writer_lock(rw_map_mutex);
            for (auto& [_, v] : tls_map)
                fn(v);
        }

        inline Tp& local()
        {
            const auto this_thread_id = std::this_thread::get_id();
            {
                std::shared_lock<mutext_type> reader_lock(rw_map_mutex);
                auto itr = tls_map.find(this_thread_id);
                if (itr != tls_map.end())
                    return itr->second;
            }
            std::lock_guard<mutext_type> writer_lock(rw_map_mutex);
            auto result = tls_map.emplace(this_thread_id, init_fn());
            assert(result.second);
            return result.first->second;
        }
    };
#else
    template < typename Tp >
    class combinable final
    {
        using init_fn_type = std::function< Tp() >;
        init_fn_type init_fn;
        Tp value;
    public:
        template <typename Function>
        inline explicit combinable(Function&& init_fun)
        : init_fn(std::forward<Function>(init_fun))
        {}

        inline combinable()
        : combinable([]() { return Tp(); }) {}

        constexpr inline combinable(const combinable&) = default;
        constexpr inline combinable(combinable&&) = default;

        constexpr inline combinable& operator=(const combinable&) = default;
        constexpr inline combinable& operator=(combinable&&) = default;
        constexpr inline void clear() {}

        template < typename BinaryFunction >
        constexpr inline const Tp& combine(BinaryFunction&&) const { return value; }
        template < typename UnaryFunction >
        constexpr inline void combine_each(UnaryFunction&& fn) { fn(value); }
        constexpr inline Tp& local() { return value; }
    };
#endif
}