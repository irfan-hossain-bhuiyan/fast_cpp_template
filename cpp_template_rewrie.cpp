
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <math.h>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>
#define PI (3.14159265358979323846)
#include <array>
#include <chrono>
#include <tuple>
#include <utility>
using i32 = int32_t;
using i64 = int64_t;
using u32 = uint32_t;
using u64 = uint64_t;
using usize = size_t;
using namespace std;
using namespace std::placeholders;

template <typename ReturnType, typename... Args>
ReturnType mem(ReturnType (*func)(Args...), Args... args) {
  static unordered_map<std::tuple<Args...>, ReturnType> memo;
  // Create a tuple from the function arguments
  std::tuple<Args...> key(args...);

  // Check if the result is already memoized
  auto it = memo.find(key);
  if (it != memo.end()) {
    return it->second;
  }

  // Otherwise, calculate and memoize the result
  ReturnType result = func(args...);
  memo[key] = result;

  return result;
}

template <typename Tuple, std::size_t N> struct TuplePrinter {
  static void print(std::ostream &os, const Tuple &t) {
    TuplePrinter<Tuple, N - 1>::print(os, t);
    os << ", " << std::get<N - 1>(t);
  }
};

template <typename Tuple> struct TuplePrinter<Tuple, 1> {
  static void print(std::ostream &os, const Tuple &t) { os << std::get<0>(t); }
};

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
template <typename T = int> T get_input() {
  T x;
  std::cin >> x;
  return x;
}

const int sign(i64 value) { return value >= 0 ? 1 : -1; }

template <size_t I = 0, typename... T>
typename enable_if<I == sizeof...(T), tuple<T...>>::type
tuple_input(const tuple<T...> dummy = tuple<T...>()) {
  return dummy;
}

template <size_t I = 0, typename... T>
    typename enable_if < I<sizeof...(T), tuple<T...>>::type
                         tuple_input(const tuple<T...> dummy = tuple<T...>()) {
  get<I>(dummy) = get_input<decltype(get<I>(dummy))>();
  return tuple_input<I + 1>(dummy);
}
template <typename... Args>
std::ostream &operator<<(std::ostream &os, const std::tuple<Args...> &t) {
  os << "(";
  TuplePrinter<decltype(t), sizeof...(Args)>::print(os, t);
  os << ")";
  return os;
}

class range;
template <typename T = int> class input_iter;
template <typename O, typename T> class map;
template <typename T> class filter;
template <typename... T> class tupleiter;
template <typename... T> class productiter;
template <typename T = int> class cycle;
template <typename T = int> class flatten;
class eq_range;
class infinite_iter;
template <typename T> struct Option;
template <typename T> class Vec;

// operator overloading
// operator overloading is going here

template <size_t I>
std::ostream &operator<<(std::ostream &os, const array<int, I> &t) {
  os << "[";
  for (int i = 0; i < I; i++) {
    os << t[i] << ",";
  }
  os << "]";
  return os;
}

template <typename T> ostream &operator<<(ostream &os, const Vec<T> v) {
  os << "[";
  for (auto x : v) {
    os << x << ",";
  }
  os << "]";
}
template <typename T> ostream &operator<<(ostream &os, const Option<T> option) {
  os << "Option<";
  if (!option.has_data()) {
    os << "None";
  } else {
    os << option.get_data();
  }
  os << ">";
  return os;
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
template <typename T> using vec2d = array<T, 2>;
template <typename T> using vec3d = array<T, 3>;
// end of operator overloading in tuple
template <typename T> struct Option {
public:
  Option() : _has_data(false) {}
  Option(T input) : data(input), _has_data(true) {}

  bool has_data() const { return _has_data; }
  T get_data() const { return data; }
  template <typename O = T> Option<O> set_data(O input) {
    if (!has_data()) {
      return Option<O>();
    }
    return Option<O>(input);
  }
  T operator*() const { return get_data(); }
  Option<T> &operator=(const Option<T> &other) = default;
  template <typename O> bool operator==(const Option<O> &other) const {
    if (!(has_data() || other.has_data())) {
      return true;
    } else if (has_data() && other.has_data()) {
      return get_data() == other.get_data();
    }
    return false;
  }

private:
  T data = T();
  bool _has_data;
};

template <typename T = int> class Myiter {
public:

        using OUTPUT=T;
  template <typename O> auto map_to(function<O(OUTPUT)> f) {
    return map<O,decltype(*this)>(*this, f);
  }

  class iterator {
  private:
    Option<T> _current_data;
    Myiter &_core;

  public:
    iterator(Option<T> current_data, Myiter &core)
        : _current_data(current_data), _core(core) {}
    T operator*() const { return *_current_data; }
    T operator++() {
      _current_data = _core.next();
      return _current_data.get_data();
    }
    bool operator!=(iterator &other) const {
      return !(_current_data == other._current_data);
    }
  };

  iterator begin() { return iterator(next(), *this); }
  iterator end() { return iterator(Option<T>(), *this); }
  virtual void reset() = 0;
  virtual Option<OUTPUT> next() = 0;
  inline Myiter<T> &foreach (function<void(T)> f) {
    while (true) {
      Option<OUTPUT> result = this->next();
      if (!result.has_data()) {
        break;
      }
      f(result.get_data());
    }
    reset();
    return *this;
  }

  T reduce(T initial_value, function<T(T, T)> f) {
    T result = initial_value;
    foreach ([&result, f](T x) { result = f(result, x); })
      ;
    return result;
  }
  T sum() {
    return reduce(0, [](T x, T y) { return x + y; });
  }
  T max() {
    return reduce(next().get_data(),
                  [](int x, int y) { return x >= y ? x : y; });
  }
  T min() {
    return reduce(next().get_data(),
                  [](int x, int y) { return x <= y ? x : y; });
  }

  bool all(function<bool(T)> f) {
    while (true) {
      auto result = this->next();
      if (!result.has_data()) {
        return true;
      } else if (!f(result.get_data())) {
        return false;
      }
    }
  }

  Option<T> any(function<bool(T)> f) {
    while (true) {
      auto result = next();
      if (!result.has_data()) {
        return Option<T>();
      } else if (f(result.get_data())) {
        return result;
      }
    }
  }
  Vec<T> collect() {
    Vec<T> result = Vec<T>();
    foreach ([&result](T x) { result.push_back(x); })
      ;
    return result;
  }

  void debug_print() {
    cout << "[";
    this->foreach ([](auto x) { cout << x << ","; });
    cout << "]\n";
  }
};

class range : public Myiter<int> {
private:
  const int _min;
  const int _max;
  const int _step;
  int ans = _min - _step;

public:
  range(int max) : _min(0), _max(max), _step(sign(max)) {}
  range(int min, int max) : _min(min), _max(max), _step(sign(max - min)) {}
  range(int min, int max, int step) : _min(min), _max(max), _step(step) {}
  Option<int> next() override {
    if (sign(_step) != sign(_max - _min) or _step == 0) {
      return Option<int>();
    }
    ans += _step;
    if (sign(_step) * ans < sign(_step) * _max) {
      return Option<int>(ans);
    }
    return Option<int>();
  }
  void reset() override { ans = _min - _step; }

  range rev() {
    const auto _new_max = _max - (_max - _min) % _step;
    return range(_new_max - _step, _min - _step, -_step);
  }
};

class infinite_iter : public Myiter<int> {
private:
  int _ans;
  int _initial_value;

public:
  infinite_iter(int initial_value = 0)
      : _ans(initial_value), _initial_value(initial_value) {}
  Option<int> next() override { return Option<int>(_ans++); }
  void reset() override { _ans = _initial_value; }
};
template <typename T, size_t I> class arrayiter : public Myiter<array<T, I>> {
private:
  array<T, I> _core;

public:
  arrayiter(array<unique_ptr<Myiter<T>>, I> arr) : _core(arr) {}
  Option<array<T, I>> next() override {
    array<T, I> ans;
    for (size_t i = 0; i < I; i++) {
      auto a = _core[i]->next();
      if (!a.has_data()) {
        return Option<array<T, I>>();
      }
      ans[i] = a.get_data();
    }
    return ans;
  }
  void reset() override {
    for (auto &x : _core) {
      x->reset();
    }
  }
};

class range_eq : public Myiter<int> {
private:
  const int _min;
  const int _max;
  const int _step;
  int ans = _min - _step;

public:
  range_eq(int min, int max, int step) : _min(min), _max(max), _step(step) {}
  range_eq(int max) : _min(0), _max(max), _step(max > 0 ? 1 : -1) {}
  range_eq(int min, int max) : _min(min), _max(max), _step(sign(max - min)) {}
  Option<int> next() override {
    if (sign(_step) != sign(_max - _min) or _step == 0) {
      return Option<int>();
    }
    ans += _step;
    if (sign(_step) * ans <= sign(_step) * _max) {
      return Option<int>(ans);
    }
    return Option<int>();
  }

  void reset() override { ans = _min - _step; }

  range_eq rev() {
    const auto _new_max = _max - (_max - _min) % _step;
    return range_eq(_new_max, _min, -_step);
  }
};

template <typename T> class input_iter : public Myiter<T> {
public:
  input_iter(int input) : _input(input) {}
  Option<T> next() override {
    if (index < _input) {
      index++;
      return Option<T>(get_input<T>());
    }
    return Option<T>();
  }
  void reset() override { index = 0; }

private:
  const int _input;
  int index = 0;
};

template <typename O,typename TITER > class map : public Myiter<O> {
public:
   using TOUTPUT = typename TITER::OUTPUT;
  map(TITER core, function<O(TOUTPUT)> f) : _core(core), _f(f) {}

  // this two function should be for every MyIter class
  Option<O> next() override {
    auto result = _core.next();
    return result.set_data(_f(result.get_data()));
  }
  void reset() override { _core.reset(); }

private:
  TITER _core;
  const function<O(TOUTPUT)> _f;
};

int main() { range(10).map_to<int>([](auto x){return x*2;}).debug_print(); }
