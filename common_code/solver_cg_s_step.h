#ifndef solver_cg_s_step_h
#define solver_cg_s_step_h

#include "../common_code/timer.h"

/**
 * S-step CG according to Algorithm 2 in
 * https://research.nvidia.com/sites/default/files/pubs/2016-04_S-Step-and-Communication-Avoiding/nvr-2016-003.pdf
 */
class SolverCGSStep
{
public:
  SolverCGSStep(SolverControl &cn, const unsigned int n_steps = 1)
    : cn(cn)
    , n_steps(n_steps)
    , times(7, 0.0)
  {}

  template <typename Operator, typename VectorType>
  void
  solve(const Operator &A, VectorType &x, const VectorType &f)
  {
    Timer time;

    std::vector<std::shared_ptr<VectorType>> T(n_steps + 1);
    std::vector<std::shared_ptr<VectorType>> P_(n_steps);

    for (auto &t : T)
      {
        t = std::make_shared<VectorType>();
        A.initialize_dof_vector(*t);
      }

    for (auto &p : P_)
      {
        p = std::make_shared<VectorType>();
        A.initialize_dof_vector(*p);
      }

    std::array<typename VectorType::value_type *, 10> P;

    for (unsigned int i = 0; i < n_steps; ++i)
      P[i] = P_[i]->begin();

    VectorType &r = *T[0];

    std::array<typename VectorType::value_type *, 10> R;
    std::array<typename VectorType::value_type *, 10> Q;
    for (unsigned int i = 0; i < n_steps; ++i)
      {
        R[i] = T[i + 0]->begin();
        Q[i] = T[i + 1]->begin();
      }

    FullMatrix<typename VectorType::value_type> C(n_steps, n_steps);
    FullMatrix<typename VectorType::value_type> W(n_steps, n_steps);
    FullMatrix<typename VectorType::value_type> B(n_steps, n_steps);

    Vector<typename VectorType::value_type>      g(n_steps);
    Vector<typename VectorType::value_type>      a(n_steps);
    std::vector<typename VectorType::value_type> temp1(n_steps, 0);
    std::vector<typename VectorType::value_type> temp2(n_steps * n_steps, 0);
    std::vector<typename VectorType::value_type> temp3(n_steps * n_steps + n_steps, 0);

    const unsigned int local_size = r.local_size();

    const auto compute_residual = [](const auto &A, const auto &x, const auto &f, auto &r) {
      A.vmult(r,
              x,
              [&](const unsigned int a, const unsigned int b) {
                auto r_ = r.begin();

                for (unsigned int k = a; k < b; ++k)
                  r_[k] = 0.0;
              },
              [&](const unsigned int a, const unsigned int b) {
                auto r_ = r.begin();
                auto f_ = f.begin();

                for (unsigned int k = a; k < b; ++k)
                  r_[k] = f_[k] - r_[k];
              });
    };


    compute_residual(A, x, f, r);

    auto conv = cn.check(0, r.l2_norm());

    unsigned int c = 1;

    times[0] += time.wall_time();

    while (conv == SolverControl::iterate)
      {
        // matrix-power kernel (r=k; w=k)
        {
          ScopedTimer timer(times[1]);

          for (unsigned int i = 0; i < n_steps; ++i)
            A.vmult(*T[i + 1], *T[i]);
        }

        // find direction
        {
          ScopedTimer timer(times[2]);

          if (c == 1)
            {
              for (unsigned int i = 0; i < n_steps; ++i)
                std::memcpy(P[i], R[i], local_size * sizeof(typename VectorType::value_type));
            }
          else
            {
              // block-dot products (r=2*k; w=0)
              {
                C = 0.0;
                for (unsigned int k = 0; k < local_size; ++k)
                  for (unsigned int i = 0; i < n_steps; ++i)
                    for (unsigned int j = 0; j < n_steps; ++j)
                      C[i][j] += -(Q[i][k] * P[j][k]);

                for (unsigned int i = 0, c = 0; i < n_steps; ++i)
                  for (unsigned int j = 0; j < n_steps; ++j, ++c)
                    temp2[c] = C[i][j];

                MPI_Allreduce(
                  MPI_IN_PLACE, temp2.data(), temp2.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                for (unsigned int i = 0, c = 0; i < n_steps; ++i)
                  for (unsigned int j = 0; j < n_steps; ++j, ++c)
                    C[i][j] = temp2[c];
              }

              // find scalars beta
              W.mmult(B, C, false);

              // update search directions (r=k+1; w=k)
              for (unsigned int k = 0; k < local_size; ++k)
                {
                  for (unsigned int i = 0; i < n_steps; ++i)
                    {
                      temp1[i] = P[0][k] * B[0][i];
                      for (unsigned int j = 1; j < n_steps; ++j)
                        temp1[i] += P[j][k] * B[j][i];
                    }

                  for (unsigned int i = 0; i < n_steps; ++i)
                    P[i][k] = R[i][k] + temp1[i];
                }
            }
        }

        // block-dot products (r=2*k; w=0)
        {
          ScopedTimer timer(times[3]);

          W = 0.0;
          for (unsigned int k = 0; k < local_size; ++k)
            for (unsigned int i = 0; i < n_steps; ++i)
              for (unsigned int j = 0; j < n_steps; ++j)
                W[i][j] += Q[i][k] * P[j][k];

          g = 0.0;
          for (unsigned int k = 0; k < local_size; ++k)
            for (unsigned int i = 0; i < n_steps; ++i)
              g[i] += P[i][k] * r.local_element(k);

          for (unsigned int i = 0, c = 0; i < n_steps; ++i)
            for (unsigned int j = 0; j < n_steps; ++j, ++c)
              temp3[c] = W[i][j];

          for (unsigned int i = 0, c = n_steps * n_steps; i < n_steps; ++i, ++c)
            temp3[c] = g[i];

          MPI_Allreduce(
            MPI_IN_PLACE, temp3.data(), temp3.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

          for (unsigned int i = 0, c = 0; i < n_steps; ++i)
            for (unsigned int j = 0; j < n_steps; ++j, ++c)
              W[i][j] = temp3[c];

          for (unsigned int i = 0, c = n_steps * n_steps; i < n_steps; ++i, ++c)
            g[i] = temp3[c];
        }

        {
          ScopedTimer timer(times[4]);

          // find scalars alpha
#ifdef FORCE_ITERATION
          W = 0.0;
          for (unsigned int j = 0; j < n_steps; ++j)
            W[j][j] = 0.0001;
#endif
          W.gauss_jordan();
          W.vmult(a, g, false);

          // compute new approximation x (r=k+1; w=1)
          for (unsigned int j = 0; j < local_size; ++j)
            for (unsigned int i = 0; i < n_steps; ++i)
              x.local_element(j) += P[i][j] * a[i];
        }

        {
          ScopedTimer timer(times[5]);

          // compute residual (r=3; w=1)
          compute_residual(A, x, f, r);
        }

        {
          ScopedTimer timer(times[6]);

          conv = cn.check(c, r.l2_norm());
        }

        ++c;
      }
  }

  const std::vector<double> &
  get_profile()
  {
    return times;
  }


private:
  SolverControl &    cn;
  const unsigned int n_steps;

  std::vector<double> times;
};

#endif
