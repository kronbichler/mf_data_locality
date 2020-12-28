#ifndef timer_h_
#define timer_h_

class ScopedTimer
{
public:
  ScopedTimer(double &result)
    : result(result)
  {}

  ~ScopedTimer()
  {
    result += time.wall_time();
  }

private:
  double &result;
  Timer   time;
};

#endif
