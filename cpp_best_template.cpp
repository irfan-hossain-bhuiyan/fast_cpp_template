// Never forget to use lambda gcd other than gcd to solve problem.
// error=SegmentTree(a,gcd),
// ok=SegmentTree(a,lambda::gcd);
#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <math.h>
#include <numeric>
#include <ostream>
#include <queue>
#include <sys/types.h>
#include <tuple>
#include <type_traits>
#include <typeindex>
#include <variant>
#include <vector>
#define PI (3.14159265358979323846)
#include <array>
#include <chrono>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
using i32 = int32_t;
using i64 = int64_t;
// using u32 = uint32_t;
// using u64 = uint64_t;
using u8 = uint8_t;
// using i64 = size_t;
template <typename T> using maxheap = std::priority_queue<T>;
template <typename T>
using minheap = std::priority_queue<T, std::vector<T>, std::greater<T>>;
template <typename T> class Vec;
template <typename T> using limit = std::numeric_limits<T>;
template <typename T, typename Enable = void> class range_iterator;
using namespace std;
using namespace std::placeholders;
// Loop macros for brevity
#define RANGE(i, a, b) for (int i = (a); i < (b); ++i)
#define RANGEB(i, a, b, s) for (i64 i = (a); i < (b) - (s) + 1; i += (s))
#define RANGES(i, a, b, s) for (i64 i = (a); i < (b); i += (s))
#define RANGEREV(i, a, b) for (int i = (b) - 1; i >= (a); --i)
#define RANGEREVS(i, a, b, s) for (i64 i = (b) - 1; i >= (a); i -= s)
#define LOOP(index, value, vector)                                             \
  for (int index = 0; index < (vector).size(); index++)                        \
    if (auto &&value = vector[index]; true)
#define LOOP2(index, value1, value2, vector1, vector2)                         \
  for (int index = 0; index < min(vector1.size(), vector2.size()); index++)    \
    if (auto &&value1 = vector1[index]; auto &&value2 = vector2[index]; true)

namespace lambda {
const auto add = [](auto x, auto y) { return x + y; };
const auto mul = [](auto x, auto y) { return x * y; };
const auto max = [](auto x, auto y) { return std::max(x, y); };
const auto min = [](auto x, auto y) { return std::min(x, y); };
const auto gcd = [](auto x, auto y) { return std::gcd(x, y); };
const auto lcm = [](auto x, auto y) { return std::lcm(x, y); };
const auto equal = [](auto x, auto y) { return x == y; };
const auto n_eq = [](auto x, auto y) { return x != y; };
} // namespace lambda
template <typename Tuple, i64 N>
typename enable_if<N == 1, void>::type __tuple_print(std::ostream &os,
                                                     const Tuple &t) {
  os << std::get<0>(t);
}

template <typename Tuple, std::size_t N>
typename enable_if<N != 1, void>::type __tuple_print(std::ostream &os,
                                                     const Tuple &t) {
  __tuple_print<Tuple, N - 1>(os, t);
  os << ", " << std::get<N - 1>(t);
}
template <typename T> struct IsTuple : std::false_type {};

template <typename... Args>
struct IsTuple<std::tuple<Args...>> : std::true_type {};
template <typename Tuple, i64 Index>
typename enable_if<Index == tuple_size<Tuple>::value, istream &>::type
__tuple_input(std::istream &is, Tuple &tuple) {
  return is;
}
template <typename Tuple, std::size_t Index = 0>
typename enable_if<Index != tuple_size<Tuple>::value, istream &>::type
__tuple_input(std::istream &is, Tuple &tuple) {
  is >> std::get<Index>(tuple);
  return __tuple_input<Tuple, Index + 1>(is, tuple);
}

template <typename... Args>
std::istream &operator>>(std::istream &is, std::tuple<Args...> &tuple) {
  return __tuple_input(is, tuple);
}
template <typename T> struct Option {
  bool has_data;
  T data;
  Option() : has_data(false), data(0) {}
  Option(T data) : data(data) {}
  T unwarp_or(T data) {
    if (has_data) {
      return this->data;
    } else {
      return data;
    }
  }
  template <typename F> T operation_with(F f, T other) {
    if (has_data) {
      return f(data, other);
    }
    return other;
  }
  template <typename F> Option<T> operation_with(F f, Option<T> other) {
    if (this->has_data && other.has_data) {
      return Option(f(this->data, other->data));
      ;
    }
    if (!(this->has_data || other.has_data)) {
      return Option();
    }
    if (this->has_data) {
      return this->data;
    }
    if (other.has_data) {
      return other.data;
    }
  }
  template <typename F> Option<T> map_to(F f) {
    if (has_data) {
      return Option<T>(f(data));
    }
    return Option();
  }
  bool operator==(Option<T> other) {
    if (has_data || other.has_data == 0)
      return true;
    if (has_data && other.has_data)
      return data == other.data;
    return false;
  }
};
template <typename T> struct Option<T *> {
  T *data;
  Option(T *ref) : data(ref) {}
  Option<T> deref() {
    if (data == NULL)
      return Option();
    return Option<T>(*data);
  }
  void set(T value) {
    if (data != NULL) {
      *data = value;
    }
  }
  template <typename F> void set(F f) {
    if (data != NULL) {
      *data = f(*data);
    }
  }
  void operator=(T value) { set(value); }
};
template <typename Iterator> struct Slice {
  using iterator_type = typename std::iterator_traits<Iterator>::value_type;
  Iterator _begin;
  Iterator _end;
  Slice(Iterator begin, Iterator end) : _begin(begin), _end(end) {}
  template <typename T>
  Slice(T collection) : Slice(collection.begin(), collection.end()) {}
  void sort() { std::sort(_begin, _end); }
  template <typename F> void sort(F f) { std::sort(_begin, _end, f); }
  template <typename F> void sort_by(F f) {
    std::sort(_begin, _end, [&f](auto x, auto y) { return f(x) < f(y); });
  }
  Slice<Iterator> binary_search(iterator_type value) {
    auto it = std::equal_range(_begin, _end);
    return Slice<Iterator>(it.first, it.second);
  }
  bool empty() { return _begin == _end; }
  template <typename F>
  Slice<Iterator> binary_search(iterator_type value, F f) {
    auto it = std::equal_range(_begin._end, f);
    return Slice<Iterator>(it.first, it.second);
  }
  template <typename F>
  Slice<Iterator> binary_search_by(iterator_type value, F f) {
    auto it = std::equal_range(_begin, _end,
                               [&f](auto x, auto y) { return f(x) < f(y); });
    return Slice<Iterator>(it.first, it.second);
  }
  i64 size() { return _end - _begin; }
  template <typename F> void transform(F f) { std::transform(_begin, _end, f); }
  iterator_type reduce(iterator_type defalut_value) {
    return std::accumulate(_begin, _end, defalut_value);
  }
  void accumulate(iterator_type defalut_value) {
    std::partial_sum(_begin, _end, _begin, defalut_value);
  }
  template <typename F> iterator_type reduce(F f, iterator_type defalut_value) {
    return std::accumulate(_begin, _end, defalut_value, f);
  }
  iterator_type max() {
    return reduce(lambda::max, numeric_limits<iterator_type>::min());
  }
  iterator_type min() {
    return reduce(lambda::min, numeric_limits<iterator_type>::max());
  }
  template <typename F>
  iterator_type accumulate(F f, iterator_type defalut_value) {
    return std::partial_sum(_begin, _end, _begin, defalut_value, f);
  }
  Iterator begin() { return _begin; }
  Iterator end() { return _end; }
  Vec<iterator_type> collect() {
    Vec<iterator_type> ans;
    std::copy(_begin, _end, std::back_inserter(ans));
    return ans;
  }
  Option<iterator_type> get(i64 index) {
    if (index < 0) {
      return Option<iterator_type>();
    }
    auto it = std::next(_begin, index);
    if (it >= _end) {
      return Option<iterator_type>();
    }
    return Option(*it);
  }
  iterator_type &at(i64 index) { return *std::next(_begin, index); }
  void reverse() { std::reverse(_begin, _end); }
};

template <typename T> Slice<typename T::reverse_iterator> rSlice(T collection) {
  return Slice(collection.rbegin(), collection.rend());
}
template <typename T> class Vec : public std::vector<T> {
public:
  using std::vector<T>::vector; // Inherit constructors
  using It = typename Vec<T>::iterator;
  using rIt = typename Vec<T>::reverse_iterator;
  // Add any additional functions or modifications as needed
  Option<T> get(i64 index) {
    if (index < 0 || index >= this->size())
      return Option<T>();
    return Option<T>(this->at(index));
  }
  Option<T *> get_ref(i64 index) {
    if (index < 0 || index >= this->size())
      return Option<T *>(nullptr);
    return Option<T *>(&this->at(index));
  }
  static Vec<T> with_capacity(i64 size) {
    auto vec = Vec<T>();
    if (size <= 0) {
      return vec;
    }
    vec.reserve(size);
    return vec;
  }
  Vec<T> range(i64 last) {
    auto vec = Vec<T>();
    vec.reserve(last);
    RANGE(i, 0, last) { vec.push_back(i); }
    return vec;
  }
  Slice<It> slice() { return Slice<It>(this->begin(), this->end()); }
  Slice<It> slicestart(i64 n) {
    return Slice(std::next(this->begin, n), this->end);
  }
  Slice<It> sliceend(i64 n) {
    return Slice(this->begin(), std::next(this->begin, n));
  }
  Slice<It> slice(i64 f, i64 l) {
    return Slice(std::next(this->begin(), f), std::next(this->begin(), l));
  }

  Slice<rIt> rslice() { return Slice<rIt>(this->rbegin(), this->rend()); }

  Slice<rIt> rslicestart(i64 n) {
    return Slice(this->rbegin(), this->rend() - n);
  }

  Slice<rIt> rsliceend(i64 n) {
    return Slice(this->rbegin() + n, this->rbegin());
  }

  Slice<rIt> rslice(i64 f, i64 l) {
    return Slice(this->rbegin() + f, this->rbegin() + l);
  }
  template <typename... Collections, typename F>
  bool all(F f, Collections... coll) {
    return all_op(f, *this, coll...);
  }

  template <typename... Collections, typename F>
  bool any(F f, Collections... coll) {
    return any_op(f, *this, coll...);
  }
  template <typename... Collections, typename F>
  void map(F f, Collections... coll) {
    mutate(f, *this, coll...);
  }
};
// Recursive template to input elements of a tuple
// Base case to stop recursion
// Helper function to call TupleInput

template <typename T> T get_input() {
  T x;
  std::cin >> x;
  return x;
}
template <typename... T> tuple<T...> tuple_input() {
  return get_input<tuple<T...>>();
}

const int sign(i64 value) { return value == 0 ? 0 : value >= 0 ? 1 : -1; }

template <typename... Args>
std::ostream &operator<<(std::ostream &os, const std::tuple<Args...> &t) {
  os << "(";
  __tuple_print<decltype(t), sizeof...(Args)>(os, t);
  os << ")";
  return os;
}

template <typename T, size_t I>
std::ostream &operator<<(std::ostream &os, const array<T, I> &t) {
  os << "[";
  for (auto x : t) {
    os << x << ",";
  }
  os << "]";
  return os;
}
template <typename T>
std::ostream &operator<<(std::ostream &os, const Vec<T> &t) {
  os << "[";
  for (auto x : t) {
    os << x << ",";
  }
  os << "]";
  return os;
}
template <typename T> Vec<T> vec_input(i64 i) {
  auto v = Vec<T>();
  v.reserve(i);
  RANGE(_, 0, i) { v.push_back(get_input<T>()); }
  return v;
}

void fast_io() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
}
template <typename T> using vec2d = array<T, 2>;
template <typename T> using vec3d = array<T, 3>;

template <typename T, typename F> struct SegmentTree {
  Vec<Vec<T>> core;
  const F f;
  SegmentTree(Vec<Vec<T>> core, F f) : core(std::move(core)), f(f) {}
  SegmentTree(Vec<T> input, F f) : f(f) {
    while (input.size() > 1) {
      Vec<T> temp;
      RANGEB(i, 0, input.size(), 2) {
        temp.push_back(f(input[i], input[i + 1]));
      }
      this->core.push_back(std::move(input));
      input = std::move(temp);
    }
    this->core.push_back(std::move(input));
  }

  T query(i64 L, i64 R, i64 level = 0) {
    tuple<bool, T> def = tuple(false, T());
    auto defset = [&def, this](auto other) {
      if (get<0>(def) == false) {
        def = tuple(true, other);
      } else {
        get<1>(def) = f(get<1>(def), other);
      }
    };
    if (L == R) {
      return core[level][R];
    }
    if (L > R) {
      return T();
    }
    if (R % 2 == 0) {
      defset(core[level][R]);
      R--;
    }
    if (L % 2 == 1) {
      defset(core[level][L]);
      L++;
    }
    defset(query(L / 2, R / 2, level + 1));
    return get<1>(def);
  }
  void change(i64 pointer, T value) {
    for (auto &x : this->core) {
      x[pointer] = value;
      auto other_pointer = pointer % 2 == 0 ? pointer + 1 : pointer - 1;
      if (other_pointer >= x.size())
        break;
      value = f(x[pointer], x[other_pointer]);
      pointer /= 2;
    }
  }
};
template <typename Key, class Val>
class HashMap : public std::unordered_map<Key, Val> {
public:
  using std::unordered_map<Key, Val>::unordered_map; // Inherit constructors
  // Add any additional functions or modifications as needed
  bool contains(Key key) { return this->find(key) != this->end(); }
};

void sol() {
  auto n = get_input<i32>();
  auto l = vec_input<i64>(n);
  i64 product = 1;
  for (auto x : l) {
    product *= sign(x);
  }
  if (product <= 0) {
    cout << "0\n";
  } else {
    cout << "1\n1 0\n";
  }
}
i64 factorial(i64 num, i64 last = 1) {
  i64 ans = 1;
  RANGE(i, last, num) ans *= i;
  return ans;
}
const tuple<Vec<i32>, Vec<i32>> prime_factorization(const i32 range) {
  Vec<i32> prime_factors;
  if (range <= 0) {
    return tuple(Vec<i32>(), Vec<i32>());
  }
  auto factors = Vec<i32>::with_capacity(range + 5);
  factors.push_back(0);
  factors.push_back(1);
  RANGE(i, 2, range + 1) {
    bool prime = true;
    for (auto x : prime_factors) {
      if (i % x == 0) {
        factors.push_back(x);
        prime = false;
        break;
      }
    }
    if (prime) {
      factors.push_back(i);
      prime_factors.push_back(i);
    }
  }
  return tuple(prime_factors, factors);
}
//const auto [primes, factors] = prime_factorization(1000);
//tuple<Vec<i64>, Vec<i64>> number_factors(i64 num) {
//  Vec<i64> ans1;
//  Vec<i64> ans2;
//  while (num != 1) {
//    i64 p = factors[num];
//    i64 n = 0;
//    while (num % p == 0) {
//      n++;
//      num /= p;
//    }
//    ans1.push_back(p);
//    ans2.push_back(n);
//  }
//  return tuple(ans1, ans2);
//}
auto identity = [](auto x) { return x; };
// template <typename T, typename... O, typename F>
// bool all(Vec<T> &&main, Vec<O> &&...sub, F f = identity) {
//   RANGE(i, 0, min(main.size(), sub.size()...)) {
//     if (f(main[i], sub[i]...)) {
//       continue;
//     }
//     return false;
//   }
//   return true;
// }
template <typename Collection, typename... Collections, typename F>
bool all_op(F f, Collection &&main, Collections &&...sub) {
  auto min_size_value = min(main.size(), sub.size()...);
  RANGE(i, 0, min_size_value) {
    if (!f(main[i], sub[i]...)) {
      return false;
    }
  }

  return true;
}
template <typename Collection, typename... Collections, typename F>
bool any_op(F f, Collection &&main, Collections &&...sub) {
  RANGE(i, 0, min(main.size(), sub.size()...)) {
    if (f(main[i], sub[i]...)) {
      return true;
    }
  }
  return false;
}
template <typename Collection> struct Custom_product_iterator {
  Collection &main_ref;
  using ValueType = typename Collection::value_type;
  Vec<ValueType> main = Vec<ValueType>(main_ref.size() + 1);
  Custom_product_iterator(Collection &main_ref) : main_ref(main_ref) {}
  Vec<ValueType> operator++() {
    RANGE(i, 0, main.size()) {
      main[i]++;
      if (main_ref.size() < i) {
        break;
      }
      if (main[i] == main_ref[i]) {
        main[i] = 0;
        continue;
      }
      break;
    }
    return main;
  }
  Custom_product_iterator<Collection> begin() {
    return Custom_product_iterator<Collection>(this->main_ref);
  }
  Custom_product_iterator<Collection> end() {
    auto ans = Custom_product_iterator<Collection>(this->main_ref);
    ans.main[main_ref.size()] = 1;
    return ans;
  }
  Vec<ValueType> operator*() { return main; }
  Vec<ValueType> &operator->() { return this->main; }
  bool operator==(Custom_product_iterator<Collection> &other) {
    return all_op(lambda::equal, this->main, other.main);
  }
  bool operator!=(Custom_product_iterator<Collection> &other) {
    return any_op(lambda::n_eq, this->main, other.main);
  }
};
// int main() {
//   Vec<i64> base = Vec<i64>{1, 2, 3, 4};
//   for (auto &&x : Custom_product_iterator(base)) {
//     cout << x << "\n";
//   }
//   Vec<i64> other = Vec<i64>{1, 2, 3, 4};
// }
