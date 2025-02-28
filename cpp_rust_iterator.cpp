
// the one thing I need to do now is for every iterator wrapper(map,filter) I
// need to change there core_iterator to its reference if i have optimization
// pproblem.
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
class range;
template <typename T = int> class input_iter;
template <typename T = int, typename O = int> class map;
template <typename T = int> class filter;
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
template <typename... Args>
std::ostream &operator<<(std::ostream &os, const std::tuple<Args...> &t) {
  os << "(";
  TuplePrinter<decltype(t), sizeof...(Args)>::print(os, t);
  os << ")";
  return os;
}

template <typename T, size_t I>
std::ostream &operator<<(std::ostream &os, const array<T, I> &t) {
  os << "[";
  for (int i = 0; i < I; i++) {
    os << t[i] << ",";
  }
  os << "]";
  return os;
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
  virtual void reset() = 0;
  virtual Option<T> next() = 0;
  inline Myiter<T> &foreach (function<void(T)> f) {
    while (true) {
      Option<T> result = this->next();
      if (!result.has_data()) {
        break;
      }
      f(result.get_data());
    }
    this->reset();
    return *this;
  }

  T reduce(T initial_value, function<T(T, T)> f) {
    T result = initial_value;
    this->foreach ([&result, f](T x) { result = f(result, x); });
    return result;
  }
  T sum() {
    return this->reduce(0, [](T x, T y) { return x + y; });
  }
  T max() {
    return this->reduce(next().get_data(),
                        [](int x, int y) { return x >= y ? x : y; });
  }
  T min() {
    return this->reduce(next().get_data(),
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
      auto result = this->next();
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

  // creating a crappy iterator.Though I don't prefer iterator in c++,I created
  // it,because lambda in c++ isn't good enough
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
  void debug_print() {
    cout << "[";
    this->foreach ([](auto x) { cout << x << ","; });
    cout << "]\n";
  }

  // My_iter to other data type conversion is going on here
  template <typename O> map<O, T> map_to(function<O(T)> f);
  filter<T> filter_to(function<bool(T)> f);
  template <typename... O> tupleiter<T, O...> tuple_with(Myiter<O> &&...other);
  template <typename... O>
  productiter<T, O...> product_with(Myiter<O> &&...other);
  cycle<T> to_cycle();
  tupleiter<int, T> enumarate();
  template <typename O> map<O, T> collection_iter(O array[]) {
    return this->map_to<O>([array](auto x) { return array[x]; });
  }
};
template <typename T> class Nulliter : public Myiter<T> {
  Option<T> next() override { return Option<T>(); }
  void reset() override {}
};
// template<>
// class Myiter<int>{
//       template<typename O>
//       map<O,int> to_vectoriter(vector<O>& vec);
// };

// template <size_t I=2,typename T>
// class window_iter:public Myiter<[>
template <typename T> ostream &operator<<(ostream &os, const Vec<T> v) {
  os << "[";
  for (auto x : v) {
    os << x << ",";
  }
  os << "]";
  return os;
}
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
// O for Output and T for type where it is input
template <typename O, typename T> class map : public Myiter<O> {
public:
  map(Myiter<T> &core, function<O(T)> f) : _core(core), _f(f) {}

  // this two function should be for every MyIter class
  Option<O> next() override {
    auto result = _core.next();
    return result.set_data(_f(result.get_data()));
  }
  void reset() override { _core.reset(); }

private:
  Myiter<T> &_core;
  const function<O(T)> _f;
};

// myiter to map converter
//

template <typename T>
template <typename O>
map<O, T> Myiter<T>::map_to(function<O(T)> f) {
  return map<O, T>(*this, f);
}

// map fincton finishes.
// template<typename T>
// class flatten:Myiter<T>{
//	private:
//		Myiter<Myiter<T>&>& _core;
//		Option<Myiter<T>>& _current_iter;
//	public:
//		flatten(Myiter<Myiter<T>&>&&
// core):_core(core),_current_iter(_core.next()){} 		Option<T>
// next(){ 			if(!_current_iter.has_data()){return
// Option<T>();} Option<T> temp=_current_iter.next();
//			if(!temp.has_data()){_current_iter(_core.next());next();}
//			return temp;
//		}
//		void reset(){_core.reset();	_current_iter(_core.next());}
// };

template <typename T> class repeat : Myiter<T> {
public:
  repeat(T input) : _data(input) {}
  Option<T> next() override { return Option<T>(_data); }

private:
  T &_data;
};

template <typename T> class filter : public Myiter<T> {
public:
  filter(Myiter<T> &core, function<bool(T)> f) : _core(core), _f(f) {}
  Option<T> next() override {
    while (true) {
      auto result = _core.next();
      if (!result.has_data()) {
        return Option<T>();
      } else if (_f(*result)) {
        return result;
      }
    }
  }
  void reset() override { _core.reset(); }

private:
  Myiter<T> &_core;
  const function<bool(T)> _f;
};

template <typename T> filter<T> Myiter<T>::filter_to(function<bool(T)> f) {
  return filter<T>(*this, f);
}

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
  array<Myiter<T> &&, I> _core;

public:
  arrayiter(array<Myiter<T> &&, I> arr) : _core(arr) {}
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
template <typename... T> class tupleiter : public Myiter<tuple<T...>> {
public:
  tupleiter(Myiter<T> &&...args)
      : _iters(args...) {} // Don't know why I did use double reference
  Option<tuple<T...>> next() override { return _next(); }
  void reset() override { _reset(); }

private:
  tuple<T...> result;
  const tuple<Myiter<T> &...> _iters;
  template <const size_t I = 0>
  inline typename enable_if<I == sizeof...(T), void>::type _reset() {}
  template <const size_t I = 0>
      inline typename enable_if < I<sizeof...(T), void>::type _reset() {
    get<I>(_iters).reset();
    _reset<I + 1>();
  }
  template <const size_t I = 0>
  inline typename enable_if<I == sizeof...(T), Option<tuple<T...>>>::type
  _next() {
    return result;
  }
  template <const size_t I = 0>
      inline typename enable_if <
      I<sizeof...(T), Option<tuple<T...>>>::type _next() {
    auto check = get<I>(_iters).next();
    if (!check.has_data()) {
      return Option<tuple<T...>>();
    }
    get<I>(result) = check.get_data();
    return _next<I + 1>();
  }
};

template <typename T>
template <typename... O>
tupleiter<T, O...> Myiter<T>::tuple_with(Myiter<O> &&...other) {
  return tupleiter<T, O...>(std::move(*this), std::move(other)...);
}

template <typename T> tupleiter<int, T> Myiter<T>::enumarate() {
  static auto inf_iter = infinite_iter();
  return tupleiter<int, T>(std::move(inf_iter),std::move(*this));
}

// need a hell lots of optimization in here.
// a lots of refactoring
// works for now.
template <typename... T> class productiter : public Myiter<tuple<T...>> {
private:
  bool initialized = false;
  tuple<Myiter<T> &...> _iters;
  tuple<T...> _result;
  template <size_t I = 0>
      inline typename enable_if < I<sizeof...(T), void>::type _reset() {
    get<I>(_iters).reset();
    _reset<I + 1>();
  }
  template <size_t I = 0>
  inline typename enable_if<I == sizeof...(T), void>::type _reset() {}
  template <size_t I>
  inline typename enable_if<I == 0, void>::type _reset_next() {}
  template <size_t I>
      inline typename enable_if < 0 <
      I &&I<sizeof...(T), void>::type _reset_next() {
    get<I - 1>(_iters).reset();
    get<I - 1>(_result) =
        get<I - 1>(_iters)
            .next()
            .get_data(); // TODO:optimization mainly need here.I will work on
                         // this later,maybe later.
    _reset_next<I - 1>();
  }
  template <size_t I = 1>
      inline typename enable_if < I<sizeof...(T), void>::type _next_iter() {
    get<I>(_result) = get<I>(_iters).next().get_data();
    _next_iter<I + 1>();
  }
  template <size_t I>
  inline typename enable_if<I == sizeof...(T), void>::type _next_iter() {}
  template <size_t I = 0>
  inline typename enable_if<I == sizeof...(T), Option<tuple<T...>>>::type
  _next() {
    return Option<tuple<T...>>();
  }
  template <size_t I = 0>
  inline typename enable_if<(I < sizeof...(T)), Option<tuple<T...>>>::type
  _next() {
    auto next_data = get<I>(_iters).next();
    if (next_data.has_data()) {
      _reset_next<I>();
      get<I>(_result) = next_data.get_data();
      return Option<tuple<T...>>(_result);
    } else {
      return _next<I + 1>();
    }
  }

public:
  productiter(Myiter<T> &&...args) : _iters(args...) { _next_iter<1>(); }
  void reset() override { _reset(); }
  Option<tuple<T...>> next() override { return _next<0>(); }
};

template <typename T>
template <typename... O>
productiter<T, O...> Myiter<T>::product_with(Myiter<O> &&...other) {
  return productiter<T, O...>(std::move(*this), std::move(other)...);
}
template <size_t I> array<u64, I> int_to_array(u64 number, array<u64, I> mod) {
  array<u64, I> ans;
  for (usize i = 0; i < I; i++) {
    ans[i] = number % mod[i];
    number /= mod[i];
  }
  return ans;
}
template <size_t I> auto range_product_iter(array<u64, I> mod) {
  u64 product = 1;
  for (auto x : mod) {
    product *= x;
  }
  return range(product).map_to<array<u64, I>>(
      [mod](auto x) { return int_to_array(x, mod); });
}

template <typename T> class cycle : public Myiter<T> {
public:
  cycle(Myiter<T> &core) : _core(core){};
  Option<T> next() override {
    auto result = _core.next();
    if (!result.has_data()) {
      _core.reset();
      return _core.next();
    }
    return result;
  }
  void reset() override { _core.reset(); }

private:
  Myiter<T> &_core;
};

template <typename T> cycle<T> Myiter<T>::to_cycle() { return cycle<T>(*this); }
template <typename T> class Vec : public vector<T> {
public:
  using vector<T>::vector;
  // Sort
  void sort() { std::sort(this->begin(), this->end()); }

  // Custom comparator sort
  template <typename Compare> void sort(Compare comp) {
    std::sort(this->begin(), this->end(), comp);
  }
//  map<T,int> iter() {
//    return range(this->size()).collection_iter(this->data());
//  }
//
//  map<T,int> iter(usize index) {
//    return range(index, this->size()).collection_iter(this->data());
//  }
  //  template <size_t I> auto window_iter() {
  //  array<map<T,usize>, I> iters;
  //  for (usize i = 0; i < I; i++) {
  //          auto a=iter(i);
  //  }
  //  return arrayiter(iters);
  // }

  // Print
};

int factorial(int n) {
  return range_eq(1, n).reduce(1, [](int x, int y) { return x * y; });
}

void fast_io() {
  ios_base::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
}

template <typename T = i64> T pow(T value, T exponent) {
  T result = 1;
  while (exponent) {
    if (exponent == 0) {
      return result;
    }
    if (exponent % 2 == 0) {
      value *= value;
      exponent /= 2;
    } else {
      result *= value;
      value *= value;
      exponent /= 2;
    }
  }
}
void do_nothing() { auto x = input_iter(get_input()).collect(); }
//  fast_io();
//  auto xc = range(7);
//   Vec<int> z = xc.map_to<int>([](auto x) { return x * 2; }).collect();
//    cout<<z;
//   cout << "range(10).tuple_with(range(5)):";
//   range(10).tuple_with<int>(range(5)).debug_print();
//   cout << "rangeproductiter([4,2]):";
//   range_product_iter<2>({4, 5}).debug_print();
//   Vec<i32> x{1, 2, 3, 4, 5, 5, 6, 7};
//   tupleiter(range(x.size()).collection_iter(x.data()),
//            range(1, x.size()).collection_iter(x.data()))
//      .debug_print();
// 
//
//    cout << "range(20).enumarate:";
//    range(2).enumarate().debug_print();
//    cout << "for(auto x:range(20).rev()){cout<<x<<\"\\n\"\n;}";
//    for (auto x : range(20).rev()) {
//      cout << x << "\n";
//    }
//
 //  auto begin = chrono::high_resolution_clock::now();
 // 	range(10000000).sum();
 // 	auto end = chrono::high_resolution_clock::now();
 //  cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() <<
 //  "\n";

 //  begin = chrono::high_resolution_clock::now();
 //  int sum = 0;
 //  for (int i = 0; i < 10000000; i++) {
 //      sum += i;
 //  }
 //  end = chrono::high_resolution_clock::now();
 //  cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() <<
 //  " ans"<<sum<<"\n";


// Output:
// range(10):[0,1,2,3,4,5,6,7,8,9,]
// range(5,10):[5,6,7,8,9,]
// range(5,10,2):[5,7,]
// range(10,0,-1):[10,9,8,7,6,5,4,3,2,1,]
// range(10,0,-2):[10,8,6,4,2,]
// range_eq(10):[0,1,2,3,4,5,6,7,8,9,10,]
// range(10).map(2*x):[0,2,4,6,8,10,12,14,16,18,]
// range(10).filter(x%3==0):[0,3,6,9,]
// range(10).tuple_with(range(3)):[(0, 0),(1, 1),(2, 2),]
//[(0, 0),(1, 0),(2, 0),(3, 0),(0, 1),(1, 1),(2, 1),(3, 1),(0, 2),(1, 2),(2,
// 2),(3, 2),(0, 3),(1, 3),(2, 3),(3, 3),(0, 4),(1, 4),(2, 4),(3, 4),]

//	cout << "range(10):";
//	range(10).debug_print();
//	cout << "range(5,10):";
//	range(5, 10).debug_print();
//	cout << "range(5,10).rev():";
//	range(5, 10).rev().debug_print();
//	cout << "range(-5,-10):";
//	range(-5, -10).debug_print();
//	cout << "range(10,-2,1):";
//	range(10, -2, 1).debug_print();
//	cout << "range(10,-2,1).rev():";
//	range(10, -2, 1).rev().debug_print();
//	cout << "range(-5,-10).rev():";
//	range(-5, -10).rev().debug_print();
//	cout << "range(5,5):";
//	range(5, 5).debug_print();
//	cout << "range_eq(-2,-7):";
//	range_eq(-2, -7).debug_print();
//	cout << "range(5,6,-1).rev():";
//	range(5, 6, -1).rev().debug_print();
//	cout << "range(10).map(2*x)";
//	range(10).map_to<int>([](auto x) {return 2 * x;}).debug_print();
//	cout << "range(10).filter_to(x%2=0):";
//	range(10).filter_to([](auto x) {return x % 2 == 0;}).debug_print();
//	cout << "range(10).tuple_with(range(5)):";
//	range(10).tuple_with<int>(range(5)).debug_print();
//	cout << "rangeproductiter([4,2]):";
//	rangeproductiter<2>({4, 2}).debug_print();
//	cout << "productiter(range(10),range(5))";
//	range(10).product_with(range(5)).debug_print();
//	cout << "range(20).enumarate:";
//	range(20).enumarate().debug_print();
//	cout <<"for(auto x:range(20).rev()){cout<<x<<\"\\n\"\n;}";
//	for(auto x:range(20).rev()){cout<<x<<"\n";}

// auto begin = chrono::high_resolution_clock::now();
//	range(10000000).sum();
//	auto end = chrono::high_resolution_clock::now();
// cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() <<
// "\n";

// begin = chrono::high_resolution_clock::now();
// int sum = 0;
// for (int i = 0; i < 10000000; i++) {
//     sum += i;
// }
// end = chrono::high_resolution_clock::now();
// cout << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << "
// ans"<<sum<<"\n";
