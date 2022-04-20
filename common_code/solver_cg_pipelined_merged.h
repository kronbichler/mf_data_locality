// ---------------------------------------------------------------------
//
// Communication hiding version of conjugate gradient method based on
// the section 3.1 of "Hiding global synchronization latency in the preconditioned
// Conjugate Gradient algorithm" by P.Ghysels, W.Vanroose
// https://doi.org/10.1016/j.parco.2013.06.001 with minimization of memory
// access
//
// ---------------------------------------------------------------------
#ifndef solver_cg_pipelined_optimized_h
#define solver_cg_pipelined_optimized_h

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector_operations_internal.h>

#include "solver_cg_pipelined.h"
#include "timer.h"


template <typename VectorType>
class SolverCGPipelinedMerged : public dealii::SolverBase<VectorType>
{
public:
  /**
   * Declare type for container size.
   */
  using size_type = dealii::types::global_dof_index;

  /**
   * Constructor.
   */
  SolverCGPipelinedMerged(dealii::SolverControl &cn)
    : dealii::SolverBase<VectorType>(cn)
    , times(2, 0.0)
  {}


  /**
   * Virtual destructor.
   */
  virtual ~SolverCGPipelinedMerged() override = default;

  /**
   * Solve the linear system $Ax=b$ for x.
   */

  template <typename MatrixType, typename PreconditionerType>
  void
  solve(const MatrixType &A, VectorType &x, const VectorType &b, const PreconditionerType &)

  {
    Assert((std::is_same<PreconditionerType, dealii::PreconditionIdentity>::value),
           ExcNotImplemented());

    using number                      = typename VectorType::value_type;
    dealii::SolverControl::State conv = dealii::SolverControl::iterate;

    // Memory allocation
    typename dealii::VectorMemory<VectorType>::Pointer z_pointer(this->memory);
    typename dealii::VectorMemory<VectorType>::Pointer s_pointer(this->memory);
    typename dealii::VectorMemory<VectorType>::Pointer p_pointer(this->memory);
    typename dealii::VectorMemory<VectorType>::Pointer r_pointer(this->memory);
    typename dealii::VectorMemory<VectorType>::Pointer w_pointer(this->memory);
    typename dealii::VectorMemory<VectorType>::Pointer q_pointer(this->memory);

    // some aliases for simpler access
    VectorType &z = *z_pointer;
    VectorType &s = *s_pointer;
    VectorType &p = *p_pointer;
    VectorType &r = *r_pointer;
    VectorType &w = *w_pointer;
    VectorType &q = *q_pointer;


    int    it  = 0;
    double res = std::numeric_limits<double>::max();

    // resize the vectors, but do not set
    // the values since they'd be overwritten
    // soon anyway.
    z.reinit(x, false);
    s.reinit(x, false);
    p.reinit(x, false);

    r.reinit(x, true);
    w.reinit(x, true);
    q.reinit(x, true);

    VectorizedArray<number> *r_vectorized = reinterpret_cast<VectorizedArray<number> *>(r.begin());
    VectorizedArray<number> *x_vectorized = reinterpret_cast<VectorizedArray<number> *>(x.begin());
    VectorizedArray<number> *w_vectorized = reinterpret_cast<VectorizedArray<number> *>(w.begin());
    VectorizedArray<number> *p_vectorized = reinterpret_cast<VectorizedArray<number> *>(p.begin());
    VectorizedArray<number> *z_vectorized = reinterpret_cast<VectorizedArray<number> *>(z.begin());
    VectorizedArray<number> *q_vectorized = reinterpret_cast<VectorizedArray<number> *>(q.begin());
    VectorizedArray<number> *s_vectorized = reinterpret_cast<VectorizedArray<number> *>(s.begin());

    number beta      = 0.;
    number gamma     = 0.;
    number alpha     = 0.;
    number gamma_old = 0.;
    number delta     = 0.;

    // compute residual. if vector is
    // zero, then short-circuit the
    // full computation
    if (!x.all_zero())
      {
        A.vmult(r, x);
        r.add(-1., b);
      }
    else
      {
        ScopedTimer timer(times[1]);
        r.equ(-1., b);
      }

    {
      ScopedTimer timer(times[1]);
      r *= -1.0;
      res = r.l2_norm();
    }

    conv = this->iteration_status(0, res, x);
    if (conv != dealii::SolverControl::iterate)
      return;

    {
      ScopedTimer timer(times[0]);
      A.vmult(w, r);
    }


    NonBlockingDotproduct<number, VectorType> nonblocking;
    // start earlier to finish in the beginning of
    // first iteration
    {
      ScopedTimer timer(times[1]);
      nonblocking.dot_product_start(r, r, &gamma);
      nonblocking.dot_product_start(w, r, &delta);
      nonblocking.dot_products_commit();
    }
    number local_dot_sum_1;
    number local_dot_sum_2;

    const typename VectorType::size_type vec_size_local = x.locally_owned_size();

    while (conv == dealii::SolverControl::iterate)
      {
        it++;


        {
          ScopedTimer timer(times[0]);
          A.vmult(q, w);
        }

        nonblocking.dot_products_finish();

        // compute status
        res  = std::sqrt(gamma);
        conv = this->iteration_status(it, res, x);

        if (conv != dealii::SolverControl::iterate)
          {
            break;
          }

        if (it == 1)
          {
            alpha = gamma / delta;
            beta  = 0.0;
          }
        else
          {
            beta  = gamma / gamma_old;
            alpha = gamma / (delta - (beta * gamma / alpha));
          }
        // vector operations merged under one loop

        local_dot_sum_1 = 0;
        local_dot_sum_2 = 0;

        unsigned int vectorized_start = 0;
        unsigned int vectorized_end =
          vec_size_local - (vec_size_local % VectorizedArray<number>::size());
        unsigned int vect_size =
          (vectorized_end - vectorized_start) / (VectorizedArray<number>::size());

        VectorizedArray<number> local_dot_sum_1_vec = VectorizedArray<number>();
        VectorizedArray<number> local_dot_sum_2_vec = VectorizedArray<number>();

        {
          ScopedTimer timer(times[1]);
          for (unsigned int i = 0; i < vect_size; ++i)
            {
              p_vectorized[i] = r_vectorized[i] + beta * p_vectorized[i];
              x_vectorized[i] = x_vectorized[i] + alpha * p_vectorized[i];
              s_vectorized[i] = w_vectorized[i] + beta * s_vectorized[i];
              r_vectorized[i] = r_vectorized[i] - alpha * s_vectorized[i];
              z_vectorized[i] = q_vectorized[i] + beta * z_vectorized[i];
              w_vectorized[i] = w_vectorized[i] - alpha * z_vectorized[i];

              local_dot_sum_1_vec += r_vectorized[i] * r_vectorized[i];
              local_dot_sum_2_vec += w_vectorized[i] * r_vectorized[i];
            }

          unsigned int nvec = VectorizedArray<number>::size();
          for (unsigned int j = 0; j < nvec; ++j)
            {
              local_dot_sum_1 += local_dot_sum_1_vec[j];
              local_dot_sum_2 += local_dot_sum_2_vec[j];
            }

          for (unsigned int i = vectorized_end; i < vec_size_local; ++i)
            {
              p.local_element(i) = r.local_element(i) + beta * p.local_element(i);
              x.local_element(i) = x.local_element(i) + alpha * p.local_element(i);
              s.local_element(i) = w.local_element(i) + beta * s.local_element(i);
              r.local_element(i) = r.local_element(i) - alpha * s.local_element(i);
              z.local_element(i) = q.local_element(i) + beta * z.local_element(i);
              w.local_element(i) = w.local_element(i) - alpha * z.local_element(i);

              local_dot_sum_1 += r.local_element(i) * r.local_element(i);
              local_dot_sum_2 += w.local_element(i) * r.local_element(i);
            }
        }

        nonblocking.dot_product_start(r, &gamma, &delta, local_dot_sum_1, local_dot_sum_2);
        nonblocking.dot_products_commit();

        gamma_old = gamma;
      }

    // in case of failure: throw exception
    if (conv != dealii::SolverControl::success)
      AssertThrow(false, dealii::SolverControl::NoConvergence(it, res));
    // otherwise exit as normal
  }
  const std::vector<double> &
  get_profile()
  {
    return times;
  }

private:
  std::vector<double> times;
};

#endif
