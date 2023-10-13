class iter {
public:
  class iterator {};
  static iter::iterator begin() { return iterator(); }
};
int main() { 
	iter::iterator var=iter::begin();
}
