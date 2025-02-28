

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
using u32 = uint32_t;
using u64 = uint64_t;
using u8 = uint8_t;
using usize = size_t;
template <typename T> using maxheap = std::priority_queue<T>;
template <typename T>
using minheap = std::priority_queue<T, std::vector<T>, std::greater<T>>;
template <typename T> class Vec;
template <typename T> using limit = std::numeric_limits<T>;
using namespace std;
using namespace std::placeholders;
#define pii pair<int, int>
#define pb push_back
#define mp make_pair
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
// Loop macros for brevity
#define range(i, a, b) for (int i = (a); i < (b); ++i)
#define rangeb(i, a, b, s) for (i64 i = (a); i < (b) - (s) + 1; i += (s))
#define ranges(i, a, b, s) for (i64 i = (a); i < (b); i += (s))
#define rangerev(i, a, b) for (int i = (b)-1; i >= (a); --i)
#define rangerevs(i, a, b, s) for (i64 i = (b)-1; i >= (a); i -= s)
const auto add = [](auto x, auto y) { return x + y; };
const auto mul = [](auto x, auto y) { return x * y; };
const auto max = [](auto x, auto y) { return std::max(x, y); };
const auto min = [](auto x, auto y) { return std::min(x, y); };
const auto gcd = [](auto x, auto y) { return std::gcd(x, y); };
const auto lcm = [](auto x, auto y) { return std::lcm(x, y); };
auto add_n(i64 n) {
  return [n](auto o) { return n + o; };
}
template <typename Tuple, usize N>
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
template <typename Tuple, usize Index>
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
  usize size() { return _end - _begin; }
  template <typename F> void transform(F f) { std::transform(_begin, _end, f); }
  iterator_type reduce() { return std::accumulate(_begin, _end); }
  void accumulate() { std::partial_sum(_begin, _end, _begin); }
  template <typename F> iterator_type reduce(F f) {
    return std::accumulate(_begin, _end, f);
  }
  template <typename F> iterator_type accumulate(F f) {
    return std::partial_sum(_begin, _end, _begin, f);
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
  iterator_type &at(usize index) { return *std::next(_begin, index); }
  void reverse() { std::reverse(_begin, _end); }

};

template <typename T> class Vec : public std::vector<T> {
public:
  using std::vector<T>::vector; // Inherit constructors
  using It = typename Vec<T>::iterator;
  using rIt = typename Vec<T>::reverse_iterator;
  // Add any additional functions or modifications as needed
  Vec<char>(string s):std::vector<T>(s.begin(),s.end()){}
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
  Slice<It> slice() { return Slice<It>(this->begin(), this->end()); }
  Slice<It> slicestart(usize n) {
    return Slice(std::next(this->begin, n), this->end);
  }
  Slice<It> sliceend(usize n) {
    return Slice(this->begin(), std::next(this->begin, n));
  }
  Slice<It> slice(usize f, usize l) {
    return Slice(std::next(this->begin(), f), std::next(this->begin(), l));
  }

  Slice<rIt> rslice() { return Slice<rIt>(this->rbegin(), this->rend()); }

  Slice<rIt> rslicestart(usize n) {
    return Slice(this->rbegin(), this->rend() - n);
  }

  Slice<rIt> rsliceend(usize n) {
    return Slice(this->rbegin() + n, this->rbegin());
  }

  Slice<rIt> rslice(usize f, usize l) {
    return Slice(this->rbegin() + f, this->rbegin() + l);
  }
  //Vec<char>(string s):std::vector<char>(make_move_iterator(s.begin()),make_move_iterator(s.end())){}
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

const int sign(i64 value) { return value >= 0 ? 1 : -1; }

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

// here are some initial coding for operation overloading
template <size_t I = 0, typename... T>
inline typename enable_if<(I == sizeof...(T)), tuple<T...>>::type
_add_(const tuple<T...> &a, const tuple<T...> &b,
      tuple<T...> dummy = tuple<T...>()) {
  return dummy;
}

template <size_t I = 0, typename... T>
inline typename enable_if<(I < sizeof...(T)), tuple<T...>>::type
_add_(const tuple<T...> &a, const tuple<T...> &b,
      tuple<T...> dummy = tuple<T...>()) {
  get<I>(dummy) = get<I>(a) + get<I>(b);
  return _add_<I + 1, T...>(a, b, dummy);
}

template <size_t I = 0, typename... T>
typename enable_if<(I == sizeof...(T)), tuple<T...>>::type
_sub_(const tuple<T...> &a, const tuple<T...> &b,
      tuple<T...> dummy = tuple<T...>()) {
  return dummy;
}

template <size_t I = 0, typename... T>
typename enable_if<(I < sizeof...(T)), tuple<T...>>::type
_sub_(const tuple<T...> &a, const tuple<T...> &b,
      tuple<T...> dummy = tuple<T...>()) {
  get<I>(dummy) = get<I>(a) - get<I>(b);
  return _add_<I + 1>(a, b, dummy);
}

template <size_t I = 0, typename... T>
typename enable_if<(I == sizeof...(T)), tuple<T...>>::type
_mul_(const tuple<T...> &a, const tuple<T...> &b,
      tuple<T...> dummy = tuple<T...>()) {
  return dummy;
}

template <size_t I = 0, typename... T>
typename enable_if<(I < sizeof...(T)), tuple<T...>>::type
_mul_(const tuple<T...> &a, const tuple<T...> &b,
      tuple<T...> dummy = tuple<T...>()) {
  get<I>(dummy) = get<I>(a) * get<I>(b);
  return _add_<I + 1>(a, b, dummy);
}

template <size_t I = 0, typename... T>
typename enable_if<(I == sizeof...(T)), tuple<T...>>::type
_negative_(const tuple<T...> &a, const tuple<T...> &b,
           tuple<T...> dummy = tuple<T...>()) {
  return dummy;
}

template <size_t I = 0, typename... T>
typename enable_if<(I < sizeof...(T)), tuple<T...>>::type
_negative_(const tuple<T...> &a, tuple<T...> dummy = tuple<T...>()) {
  get<I>(dummy) = -get<I>(a);
  return _add_<I + 1>(a, dummy);
}

// the over layer code
template <typename... T>
tuple<T...> operator+(const tuple<T...> &a, const tuple<T...> &b) {
  return _add_(a, b);
}

template <typename... T>
tuple<T...> operator-(const tuple<T...> &a, const tuple<T...> &b) {
  return _sub_(a, b);
}

template <typename... T>
tuple<T...> operator*(const tuple<T...> &a, const tuple<T...> &b) {
  return _mul_(a, b);
}

template <typename... T> tuple<T...> operator-(const tuple<T...> &a) {
  return _negative_(a);
}

template <typename T, size_t N>
std::array<T, N> operator+(const std::array<T, N> &arr1,
                           const std::array<T, N> &arr2) {
  std::array<int, N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = arr1[i] + arr2[i];
  }
  return result;
}

template <typename T, size_t N>
std::array<T, N> operator-(const std::array<T, N> &arr1,
                           const std::array<T, N> &arr2) {
  std::array<int, N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = arr1[i] - arr2[i];
  }
  return result;
}

template <typename T, size_t N>
std::array<int, N> operator*(const std::array<T, N> &arr1,
                             const std::array<T, N> &arr2) {
  std::array<int, N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = arr1[i] * arr2[i];
  }
  return result;
}

template <typename T, size_t N>
std::array<int, N> operator/(const std::array<T, N> &arr1,
                             const std::array<T, N> &arr2) {
  std::array<int, N> result;
  for (size_t i = 0; i < N; ++i) {
    if (arr2[i] == 0) {
      throw std::runtime_error("Division by zero is not allowed.");
    }
    result[i] = arr1[i] / arr2[i];
  }
  return result;
}
template <typename T, size_t N>
std::array<T, N> operator-(const std::array<T, N> &arr) {
  std::array<T, N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = -arr[i];
  }
  return result;
}
template <typename T, size_t N>
std::array<T, N> &operator+=(std::array<T, N> &arr1,
                             const std::array<T, N> &arr2) {
  for (size_t i = 0; i < N; ++i) {
    arr1[i] += arr2[i];
  }
  return arr1;
}

template <typename T, size_t N>
std::array<T, N> &operator-=(std::array<T, N> &arr1,
                             const std::array<T, N> &arr2) {
  for (size_t i = 0; i < N; ++i) {
    arr1[i] -= arr2[i];
  }
  return arr1;
}

template <typename T, size_t N>
std::array<T, N> &operator*=(std::array<T, N> &arr1,
                             const std::array<T, N> &arr2) {
  for (size_t i = 0; i < N; ++i) {
    arr1[i] *= arr2[i];
  }
  return arr1;
}

template <typename T, size_t N>
std::array<T, N> &operator/=(std::array<T, N> &arr1,
                             const std::array<T, N> &arr2) {
  for (size_t i = 0; i < N; ++i) {
    if (arr2[i] == 0) {
      throw std::runtime_error("Division by zero is not allowed.");
    }
    arr1[i] /= arr2[i];
  }
  return arr1;
}
template <typename T> Vec<T> operator+(const Vec<T> &vec1, const Vec<T> &vec2) {
  Vec<T> result(vec1.size());
  for (size_t i = 0; i < vec1.size(); ++i) {
    result[i] = vec1[i] + vec2[i];
  }
  return result;
}
template<typename T> bool operator==(const Vec<T> &vec1,const Vec<T> &vec2){
	if(vec1.size!=vec2.size)return false;
	range(i,0,vec1.size()){if(vec1[i]!=vec2[i])return false;}
	return true;
}
template <typename T> Vec<T> operator-(const Vec<T> &vec1, const Vec<T> &vec2) {
  Vec<T> result(vec1.size());
  for (size_t i = 0; i < vec1.size(); ++i) {
    result[i] = vec1[i] - vec2[i];
  }
  return result;
}

template <typename T> Vec<T> operator*(const Vec<T> &vec1, const Vec<T> &vec2) {
  Vec<T> result(vec1.size());
  for (size_t i = 0; i < vec1.size(); ++i) {
    result[i] = vec1[i] * vec2[i];
  }
  return result;
}

template <typename T> Vec<T> operator/(const Vec<T> &vec1, const Vec<T> &vec2) {
  Vec<T> result(vec1.size());
  for (size_t i = 0; i < vec1.size(); ++i) {
    if (vec2[i] == 0) {
      throw std::runtime_error("Division by zero is not allowed.");
    }
    result[i] = vec1[i] / vec2[i];
  }
  return result;
}

template <typename T> Vec<T> operator-(const Vec<T> &vec) {
  Vec<T> result(vec.size());
  for (size_t i = 0; i < vec.size(); ++i) {
    result[i] = -vec[i];
  }
  return result;
}

template <typename T> Vec<T> &operator+=(Vec<T> &vec1, const Vec<T> &vec2) {
  for (size_t i = 0; i < vec1.size(); ++i) {
    vec1[i] += vec2[i];
  }
  return vec1;
}

template <typename T> Vec<T> &operator-=(Vec<T> &vec1, const Vec<T> &vec2) {
  for (size_t i = 0; i < vec1.size(); ++i) {
    vec1[i] -= vec2[i];
  }
  return vec1;
}

template <typename T> Vec<T> &operator*=(Vec<T> &vec1, const Vec<T> &vec2) {
  for (size_t i = 0; i < vec1.size(); ++i) {
    vec1[i] *= vec2[i];
  }
  return vec1;
}

template <typename T> Vec<T> &operator/=(Vec<T> &vec1, const Vec<T> &vec2) {
  for (size_t i = 0; i < vec1.size(); ++i) {
    if (vec2[i] == 0) {
      throw std::runtime_error("Division by zero is not allowed.");
    }
    vec1[i] /= vec2[i];
  }
  return vec1;
}

template <typename T, size_t N>
std::array<T, N> operator+(const std::array<T, N> &arr, const T &scalar) {
  std::array<T, N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = arr[i] + scalar;
  }
  return result;
}

template <typename T, size_t N>
std::array<T, N> operator-(const std::array<T, N> &arr, const T &scalar) {
  std::array<T, N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = arr[i] - scalar;
  }
  return result;
}

template <typename T, size_t N>
std::array<T, N> operator*(const std::array<T, N> &arr, const T &scalar) {
  std::array<T, N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = arr[i] * scalar;
  }
  return result;
}

template <typename T, size_t N>
std::array<T, N> operator/(const std::array<T, N> &arr, const T &scalar) {
  std::array<T, N> result;
  if (scalar == 0) {
    throw std::runtime_error("Division by zero is not allowed.");
  }
  for (size_t i = 0; i < N; ++i) {
    result[i] = arr[i] / scalar;
  }
  return result;
}

template <typename T, size_t N>
std::array<T, N> operator+(const T &scalar, const std::array<T, N> &arr) {
  return arr + scalar; // Commutative property
}

template <typename T, size_t N>
std::array<T, N> operator-(const T &scalar, const std::array<T, N> &arr) {
  std::array<T, N> result;
  for (size_t i = 0; i < N; ++i) {
    result[i] = scalar - arr[i];
  }
  return result;
}

template <typename T, size_t N>
std::array<T, N> operator*(const T &scalar, const std::array<T, N> &arr) {
  return arr * scalar; // Commutative property
}

template <typename T, size_t N>
std::array<T, N> operator/(const T &scalar, const std::array<T, N> &arr) {
  std::array<T, N> result;
  for (size_t i = 0; i < N; ++i) {
    if (arr[i] == 0) {
      throw std::runtime_error("Division by zero is not allowed.");
    }
    result[i] = scalar / arr[i];
  }
  return result;
}
template <typename T> Vec<T> operator+(const Vec<T> &vec1, const T &scalar) {
  Vec<T> result(vec1.size());
  for (size_t i = 0; i < vec1.size(); ++i) {
    result[i] = vec1[i] + scalar;
  }
  return result;
}
template <typename T> Vec<T> operator+(const T &scalar, const Vec<T> &vec1) {
  return vec1 + scalar;
}

template <typename T> Vec<T> operator-(const Vec<T> &vec1, const T &scalar) {
  Vec<T> result(vec1.size());
  for (size_t i = 0; i < vec1.size(); ++i) {
    result[i] = vec1[i] - scalar;
  }
  return result;
}

template <typename T> Vec<T> operator*(const Vec<T> &vec1, const T &scalar) {
  Vec<T> result(vec1.size());
  for (size_t i = 0; i < vec1.size(); ++i) {
    result[i] = vec1[i] * scalar;
  }
  return result;
}

template <typename T> Vec<T> operator/(const Vec<T> &vec1, const T &scalar) {
  Vec<T> result(vec1.size());
  for (size_t i = 0; i < vec1.size(); ++i) {
    if (scalar == 0) {
      throw std::runtime_error("Division by zero is not allowed.");
    }
    result[i] = vec1[i] / scalar;
  }
  return result;
}

template <typename T> Vec<T> &operator+=(Vec<T> &vec1, const T &scalar) {
  for (size_t i = 0; i < vec1.size(); ++i) {
    vec1[i] += scalar;
  }
  return vec1;
}

template <typename T> Vec<T> &operator-=(Vec<T> &vec1, const T &scalar) {
  for (size_t i = 0; i < vec1.size(); ++i) {
    vec1[i] -= scalar;
  }
  return vec1;
}

template <typename T> Vec<T> &operator*=(Vec<T> &vec1, const T &scalar) {
  for (size_t i = 0; i < vec1.size(); ++i) {
    vec1[i] *= scalar;
  }
  return vec1;
}

template <typename T> Vec<T> &operator/=(Vec<T> &vec1, const T &scalar) {
  for (size_t i = 0; i < vec1.size(); ++i) {
    if (scalar == 0) {
      throw std::runtime_error("Division by zero is not allowed.");
    }
    vec1[i] /= scalar;
  }
  return vec1;
}

template <typename T> struct Matrix {
  Vec<Vec<T>> data;

  Matrix(Vec<Vec<T>> elements) : data(elements) {}
  Matrix(usize rows, usize coloum, T default_value = 0)
      : data(Vec<Vec<T>>(rows, Vec<T>(coloum, default_value))) {}
  // Addition of two matrices
  Matrix<T> operator+(const Matrix<T> &other) const {
    Vec<Vec<T>> result(data.size(), Vec<T>(data[0].size()));
    for (size_t i = 0; i < data.size(); ++i) {
      for (size_t j = 0; j < data[0].size(); ++j) {
        result[i][j] = data[i][j] + other.data[i][j];
      }
    }
    return Matrix<T>(result);
  }

  // Subtraction of two matrices
  Matrix<T> operator-(const Matrix<T> &other) const {
    Vec<Vec<T>> result(data.size(), Vec<T>(data[0].size()));
    for (size_t i = 0; i < data.size(); ++i) {
      for (size_t j = 0; j < data[0].size(); ++j) {
        result[i][j] = data[i][j] - other.data[i][j];
      }
    }
    return Matrix<T>(result);
  }

  // Scalar multiplication of a matrix
  Matrix<T> operator*(const T &scalar) const {
    Vec<Vec<T>> result(data.size(), Vec<T>(data[0].size()));
    for (size_t i = 0; i < data.size(); ++i) {
      for (size_t j = 0; j < data[0].size(); ++j) {
        result[i][j] = data[i][j] * scalar;
      }
    }
    return Matrix<T>(result);
  }

  // Matrix multiplication
  Matrix<T> operator*(const Matrix<T> &other) const {
    if (data[0].size() != other.data.size()) {
      throw std::invalid_argument("Incompatible matrix dimensions");
    }
    Vec<Vec<T>> result(data.size(), Vec<T>(other.data[0].size()));
    for (size_t i = 0; i < data.size(); ++i) {
      for (size_t j = 0; j < other.data[0].size(); ++j) {
        T sum = 0;
        for (size_t k = 0; k < data[0].size(); ++k) {
          sum += data[i][k] * other.data[k][j];
        }
        result[i][j] = sum;
      }
    }
    return Matrix<T>(result);
  }

  // Display the matrix
  void display() const {
    for (const auto &row : data) {
      for (const auto &elem : row) {
        std::cout << elem << " ";
      }
      std::cout << std::endl;
    }
  }
  Vec<T> operator[](usize i) { return data[i]; }
};
// Define the memoization table

// Function to perform memoization
// template <typename ReturnType, typename... Args>
// ReturnType mem(ReturnType (*func)(Args...), Args... args) {
//  static unordered_map<std::tuple<Args...>, ReturnType> memo;
//  // Create a tuple from the function arguments
//  std::tuple<Args...> key(args...);
//
//  // Check if the result is already memoized
//  auto it = memo.find(key);
//  if (it != memo.end()) {
//    return it->second;
//  }
//
//  // Otherwise, calculate and memoize the result
//  ReturnType result = func(args...);
//  memo[key] = result;
//
//  return result;
//}
// Example usage:
// Define the function you want to memoize
template <typename T> Vec<T> vec_input(usize i) {
  auto v = Vec<T>();
  v.reserve(i);
  range(_, 0, i) { v.push_back(get_input<T>()); }
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
      rangeb(i, 0, input.size(), 2) {
        temp.push_back(f(input[i], input[i + 1]));
      }
      this->core.push_back(std::move(input));
      input = std::move(temp);
    }
    this->core.push_back(std::move(input));
  }

  T query(usize L, usize R, usize level = 0) {
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
  void change(usize pointer, T value) {
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
void sol(){
	auto a1=get_input<string>();
	auto a=Vec<char>(a1);

	
}
int main(){
	auto n=get_input<i32>();
	range(_,0,n){
		sol();
	}
}
