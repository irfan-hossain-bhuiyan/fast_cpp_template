#include <tuple>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <ostream>
#include <sys/types.h>
#include <type_traits>
#include <typeindex>
#include <vector>
#include <math.h>
#include <algorithm>
#define PI (3.14159265358979323846)
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <array>
#include <chrono>
using i32=int32_t;using i64 = int64_t;using u32=uint32_t;using u64=uint64_t;using u8=uint8_t;
using usize=size_t;
template <typename T>
using Vec=std::vector<T>;
using namespace std;
using namespace std::placeholders;
#define vi vector<int>
#define pii pair<int, int>
#define pb push_back
#define mp make_pair
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
// Loop macros for brevity
#define range(i, a, b) for (int i = (a); i < (b); ++i)
#define rangerev(i, a, b) for (int i = (b) - 1; i >= (a); --i)
template <typename Tuple,usize N>
typename enable_if<N==1,void>::type __tuple_print(std::ostream &os, const Tuple &t) { os << std::get<0>(t); }

template <typename Tuple, std::size_t N> 
typename enable_if<N!=1,void>::type __tuple_print(std::ostream &os, const Tuple &t) {
    __tuple_print<Tuple,N-1>(os, t);
    os << ", " << std::get<N - 1>(t);
  }
template <typename T>
struct IsTuple : std::false_type {};

template <typename... Args>
struct IsTuple<std::tuple<Args...>> : std::true_type {};
template <typename Tuple,usize Index>
typename enable_if<Index==tuple_size<Tuple>::value,istream&>::type __tuple_input(std::istream& is, Tuple& tuple)
{return is;}
template <typename Tuple, std::size_t Index = 0>
typename enable_if<Index!=tuple_size<Tuple>::value,istream&>::type __tuple_input(std::istream& is, Tuple& tuple) {
        is >> std::get<Index>(tuple);
        return __tuple_input<Tuple,Index+1>(is, tuple);
}



template <typename... Args>
std::istream& operator>>(std::istream& is, std::tuple<Args...>& tuple) {
    return __tuple_input(is, tuple);
}

// Recursive template to input elements of a tuple
// Base case to stop recursion
// Helper function to call TupleInput

template <typename T> T get_input() {
  T x;
  std::cin >> x;
  return x;
}
template <typename... T>tuple<T...> tuple_input(){
	return get_input<tuple<T...>>();
}

const int sign(i64 value) { return value >= 0 ? 1 : -1; }

template <typename... Args>
std::ostream &operator<<(std::ostream &os, const std::tuple<Args...> &t) {
  os << "(";
  __tuple_print<decltype(t),sizeof...(Args)>(os, t);
  os << ")";
  return os;
}

template <size_t I>
std::ostream &operator<<(std::ostream &os, const array<int, I> &t) {
  os << "[";
  for (int i = 0; i < I; i++) {
    os << t[i] << ",";
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


template<typename T,size_t N>
std::array<T, N> operator+(const std::array<T, N>& arr1, const std::array<T, N>& arr2) {
    std::array<int, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = arr1[i] + arr2[i];
    }
    return result;
}

template<typename T,size_t N>
std::array<T, N> operator-(const std::array<T, N>& arr1, const std::array<T, N>& arr2) {
    std::array<int, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = arr1[i] - arr2[i];
    }
    return result;
}

template<typename T,size_t N>
std::array<int, N> operator*(const std::array<T, N>& arr1, const std::array<T, N>& arr2) {
    std::array<int, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = arr1[i] * arr2[i];
    }
    return result;
}

template<typename T,size_t N>
std::array<int, N> operator/(const std::array<T, N>& arr1, const std::array<T, N>& arr2) {
    std::array<int, N> result;
    for (size_t i = 0; i < N; ++i) {
        if (arr2[i] == 0) {
            throw std::runtime_error("Division by zero is not allowed.");
        }
        result[i] = arr1[i] / arr2[i];
    }
    return result;
}
template<typename T, size_t N>
std::array<T, N> operator-(const std::array<T, N>& arr) {
    std::array<T, N> result;
    for (size_t i = 0; i < N; ++i) {
        result[i] = -arr[i];
    }
    return result;
}

// Define the memoization table

// Function to perform memoization
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
int add(int a, int b) {
    return a + b;
}
// Example usage:
// Define the function you want to memoize
template<typename T>
Vec<T> vec_input(usize i){
	auto v=Vec<T>();
	v.reserve(i);
	range(_,0,i){v.push_back(get_input<T>());}
	return v;
} 

void fast_io()
{
	ios_base::sync_with_stdio(0);
	cin.tie(0);
	cout.tie(0);
}
template<typename T>
using vec2d=array<T, 2>;
template<typename T>
using vec3d=array<T,3>;


int main(int argc, char *argv[])
{
	fast_io();
}

