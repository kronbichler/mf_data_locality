#ifndef solver_cg_s_step_h
#define solver_cg_s_step_h

#include "../common_code/timer.h"

/**
 * S-step CG according to Algorithm 2 in
 * https://research.nvidia.com/sites/default/files/pubs/2016-04_S-Step-and-Communication-Avoiding/nvr-2016-003.pdf
 */
template <unsigned int n_steps = 1>
class SolverCGSStep
{
public:
  SolverCGSStep(SolverControl &cn)
    : cn(cn)
    , times(5, 0.0)
  {}

  template <typename Operator, typename VectorType>
  void
  solve(const Operator &A, VectorType &x, const VectorType &f)
  {
    std::vector<std::shared_ptr<VectorType>> T(n_steps + 1);
    std::vector<std::shared_ptr<VectorType>> P_(n_steps);
    using Number = typename VectorType::value_type;

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

    std::array<Number *, 10> P;

    for (unsigned int i = 0; i < n_steps; ++i)
      P[i] = P_[i]->begin();

    VectorType &r = *T[0];

    std::array<Number *, 10> R;
    std::array<Number *, 10> Q;
    for (unsigned int i = 0; i < n_steps; ++i)
      {
        R[i] = T[i + 0]->begin();
        Q[i] = T[i + 1]->begin();
      }

    FullMatrix<Number> C(n_steps, n_steps);
    FullMatrix<Number> W(n_steps, n_steps);
    FullMatrix<Number> B(n_steps, n_steps);

    Vector<Number>      g(n_steps);
    Vector<Number>      a(n_steps);
    std::vector<Number> temp1(n_steps, 0);
    std::vector<Number> temp3(n_steps * n_steps + n_steps, 0);

    const unsigned int local_size = r.locally_owned_size();

    const auto compute_residual = [](const auto &A, const auto &x, const auto &f, auto &r) {
      Number residual_norm_square = 0.;
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

                Number local_sum = 0;
                for (unsigned int k = a; k < b; ++k)
                  {
                    r_[k] = f_[k] - r_[k];
                    local_sum += r_[k] * r_[k];
                  }
                residual_norm_square += local_sum;
              });
      return Utilities::MPI::sum(residual_norm_square, x.get_mpi_communicator());
    };



    auto conv = cn.check(0, std::sqrt(compute_residual(A, x, f, r)));

    unsigned int c = 1;

    while (conv == SolverControl::iterate)
      {
        // matrix-power kernel (r=k; w=k)
        {
          ScopedTimer timer(times[0]);

          for (unsigned int i = 0; i < n_steps; ++i)
            A.vmult(*T[i + 1], *T[i]);
        }

        // find direction
        {
          ScopedTimer timer(times[1]);

          if (c == 1)
            {
              for (unsigned int i = 0; i < n_steps; ++i)
                std::memcpy(P[i], R[i], local_size * sizeof(Number));
            }
          else
            {
              // block-dot products (r=2*k; w=0)
              {
                C = 0.0;
                VectorizedArray<Number> tmp[n_steps][n_steps], tmp_q[n_steps], tmp_p[n_steps];
                for (unsigned int i = 0; i < n_steps; ++i)
                  for (unsigned int j = 0; j < n_steps; ++j)
                    tmp[i][j] = 0.;
                constexpr unsigned int n_lanes = VectorizedArray<Number>::size();
                for (unsigned int k = 0; k < local_size / n_lanes; ++k)
                  {
                    for (unsigned int i = 0; i < n_steps; ++i)
                      {
                        tmp_q[i].load(Q[i] + k * n_lanes);
                        tmp_p[i].load(P[i] + k * n_lanes);
                      }
                    for (unsigned int i = 0; i < n_steps; ++i)
                      for (unsigned int j = 0; j < n_steps; ++j)
                        tmp[i][j] -= (tmp_q[i] * tmp_p[j]);
                  }
                for (unsigned int i = 0; i < n_steps; ++i)
                  for (unsigned int j = 0; j < n_steps; ++j)
                    for (unsigned int v = 0; v < n_lanes; ++v)
                      C[i][j] += tmp[i][j][v];
                for (unsigned int k = local_size / n_lanes * n_lanes; k < local_size; ++k)
                  for (unsigned int i = 0; i < n_steps; ++i)
                    for (unsigned int j = 0; j < n_steps; ++j)
                      C[i][j] -= Q[i][k] * P[j][k];

                MPI_Allreduce(
                  MPI_IN_PLACE, &C(0, 0), n_steps * n_steps, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
              }

              // find scalars beta
              W.mmult(B, C, false);
            }
        }

        // update search directions (r=k+1; w=k) and block-dot products (r=2*k; w=0)
        {
          ScopedTimer timer(times[2]);

          VectorizedArray<Number> tmp[n_steps][n_steps], tmp_q[n_steps], tmp_p[n_steps],
            tmp_r[n_steps], tmp1[n_steps], tmp_g[n_steps];
          for (unsigned int i = 0; i < n_steps; ++i)
            {
              for (unsigned int j = 0; j < n_steps; ++j)
                tmp[i][j] = 0.;
              tmp_g[i] = 0.;
            }
          constexpr unsigned int n_lanes = VectorizedArray<Number>::size();
          for (unsigned int k = 0; k < local_size / n_lanes; ++k)
            {
              for (unsigned int i = 0; i < n_steps; ++i)
                {
                  tmp_q[i].load(Q[i] + k * n_lanes);
                  tmp_p[i].load(P[i] + k * n_lanes);
                  tmp_r[i].load(R[i] + k * n_lanes);
                }
              if (c > 1)
                {
                  for (unsigned int i = 0; i < n_steps; ++i)
                    {
                      tmp1[i] = tmp_p[0] * B[0][i];
                      for (unsigned int j = 1; j < n_steps; ++j)
                        tmp1[i] += tmp_p[j] * B[j][i];
                    }

                  for (unsigned int i = 0; i < n_steps; ++i)
                    {
                      tmp_p[i] = tmp_r[i] + tmp1[i];
                      tmp_p[i].store(P[i] + k * n_lanes);
                    }
                }

              for (unsigned int i = 0; i < n_steps; ++i)
                for (unsigned int j = 0; j < n_steps; ++j)
                  tmp[i][j] += tmp_q[i] * tmp_p[j];

              VectorizedArray<Number> rv;
              rv.load(r.begin() + k * n_lanes);
              for (unsigned int i = 0; i < n_steps; ++i)
                tmp_g[i] += tmp_p[i] * rv;
            }

          for (unsigned int i = 0, c = 0; i < n_steps; ++i)
            for (unsigned int j = 0; j < n_steps; ++j, ++c)
              {
                temp3[c] = tmp[i][j][0];
                for (unsigned int v = 1; v < n_lanes; ++v)
                  temp3[c] += tmp[i][j][v];
              }

          for (unsigned int i = 0, c = n_steps * n_steps; i < n_steps; ++i, ++c)
            {
              temp3[c] = tmp_g[i][0];
              for (unsigned int v = 1; v < n_lanes; ++v)
                temp3[c] += tmp_g[i][v];
            }

          for (unsigned int k = local_size / n_lanes * n_lanes; k < local_size; ++k)
            {
              if (c > 1)
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

              for (unsigned int i = 0; i < n_steps; ++i)
                for (unsigned int j = 0; j < n_steps; ++j)
                  temp3[i * n_steps + j] += Q[i][k] * P[j][k];

              for (unsigned int i = 0; i < n_steps; ++i)
                temp3[n_steps * n_steps + i] += P[i][k] * r.local_element(k);
            }

          MPI_Allreduce(
            MPI_IN_PLACE, temp3.data(), temp3.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

          for (unsigned int i = 0, c = 0; i < n_steps; ++i)
            for (unsigned int j = 0; j < n_steps; ++j, ++c)
              W[i][j] = temp3[c];

          for (unsigned int i = 0, c = n_steps * n_steps; i < n_steps; ++i, ++c)
            g[i] = temp3[c];
        }

        {
          ScopedTimer timer(times[3]);

          // find scalars alpha
#ifdef FORCE_ITERATION
          W = 0.0;
          for (unsigned int j = 0; j < n_steps; ++j)
            W[j][j] = 0.0001;
#endif
          W.gauss_jordan();
          W.vmult(a, g, false);

          // compute new approximation x (r=k+1; w=1)
          constexpr unsigned int n_lanes = VectorizedArray<Number>::size();
          for (unsigned int k = 0; k < local_size / n_lanes; ++k)
            {
              VectorizedArray<Number> xv;
              xv.load(x.begin() + k * n_lanes);
              for (unsigned int i = 0; i < n_steps; ++i)
                {
                  VectorizedArray<Number> pi;
                  pi.load(P[i] + k * n_lanes);
                  xv += pi * a[i];
                }
              xv.store(x.begin() + k * n_lanes);
            }
          for (unsigned int k = local_size / n_lanes * n_lanes; k < local_size; ++k)
            for (unsigned int i = 0; i < n_steps; ++i)
              x.local_element(k) += P[i][k] * a[i];
        }

        {
          ScopedTimer timer(times[4]);

          // compute residual (r=3; w=1)
          const double residual_norm_square = compute_residual(A, x, f, r);

          conv = cn.check(c, std::sqrt(residual_norm_square));
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
  SolverControl &cn;

  std::vector<double> times;
};

#endif
