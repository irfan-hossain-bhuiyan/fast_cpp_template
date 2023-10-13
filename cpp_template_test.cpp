#include "cpp_template.cpp"
int main()
{
  fast_io();
  cout<<"auto xc = range(7);\nVec<int> z = xc.map_to<int>([](auto x) { return x * 2; }).collect();\ncout<<z;";
  auto xc = range(7);
   Vec<int> z = xc.map_to<int>([](auto x) { return x * 2; }).collect();
    cout<<z<<endl;
   cout << "range(10).tuple_with(range(5)):";
   range(10).tuple_with<int>(range(5)).debug_print();
   cout << "rangeproductiter([4,2]):";
   range_product_iter<2>({4, 5}).debug_print();
   cout << " Vec<i32> x{1, 2, 3, 4, 5, 5, 6, 7};\ntupleiter(range(x.size()).collection_iter(x.data()),range(1, x.size()).collection_iter(x.data())).debug_print();";
   Vec<i32> x{1, 2, 3, 4, 5, 5, 6, 7};
   tupleiter(range(x.size()).collection_iter(x.data()),range(1, x.size()).collection_iter(x.data())).debug_print();
    cout << "range(20).enumarate:";
    range(20).enumarate().debug_print();
    cout << "for(auto x:range(20).rev()){cout<<x<<\" \";}\n";
    for (auto x : range(20).rev()) {
      cout << x << " ";
    }
}
