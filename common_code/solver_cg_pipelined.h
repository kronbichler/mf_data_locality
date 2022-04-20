// ---------------------------------------------------------------------
//
// Communication hiding version of conjugate gradient method based on
// the section 3.1 of "Hiding global synchronization latency in the preconditioned
// Conjugate Gradient algorithm" by P.Ghysels, W.Vanroose
// https://doi.org/10.1016/j.parco.2013.06.001
//
// ---------------------------------------------------------------------
#ifndef solver_cg_pipelined_h
#define solver_cg_pipelined_h

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector_operations_internal.h>

#include "timer.h"

inline MPI_Datatype
mpi_type_id(const float *)
{
  return MPI_FLOAT;
}

inline MPI_Datatype
mpi_type_id(const double *)
{
  return MPI_DOUBLE;
}


inline MPI_Datatype
mpi_type_id(const long double *)
{
  return MPI_LONG_DOUBLE;
}

template <typename T>
void
iall_reduce(const MPI_Op &                    mpi_op,
            const dealii::ArrayView<const T> &values,
            const MPI_Comm &                  mpi_communicator,
            MPI_Request *                     mpi_request,
            const dealii::ArrayView<T> &      output)
{
  AssertDimension(values.size(), output.size());
  const int ierr =
    MPI_Iallreduce(values != output ? const_cast<void *>(static_cast<const void *>(values.data())) :
                                      MPI_IN_PLACE,
                   static_cast<void *>(output.data()),
                   static_cast<int>(values.size()),
                   mpi_type_id(values.data()),
                   mpi_op,
                   mpi_communicator,
                   mpi_request);
  AssertThrowMPI(ierr);
}

template <typename T>
void
isum(const dealii::ArrayView<const T> &values,
     const MPI_Comm &                  mpi_communicator,
     MPI_Request *                     mpi_request,
     const dealii::ArrayView<T> &      sums)
{
  iall_reduce(MPI_SUM, values, mpi_communicator, mpi_request, sums);
}


template <typename Number, typename VectorType>
class NonBlockingDotproduct
{
public:
  void
  dot_product_start(VectorType &a, VectorType &b, Number *result)
  {
    auto partitioner = a.get_partitioner();
    mpi_comm         = partitioner->get_mpi_communicator();

    std::shared_ptr<::dealii::parallel::internal::TBBPartitioner> dummy =
      std::make_shared<::dealii::parallel::internal::TBBPartitioner>();
    Number local_result;

    dealii::internal::VectorOperations::Dot<Number, Number> dot(a.begin(), b.begin());
    dealii::internal::VectorOperations::parallel_reduce(
      dot, 0, partitioner->locally_owned_size(), local_result, dummy);

    local_sums.push_back(local_result);
    results_queue.push_back(result);
  }

  void
  dot_product_start(VectorType &a,
                    Number *    res_1,
                    Number *    res_2,
                    Number      local_result_1,
                    Number      local_result_2)
  {
    auto partitioner = a.get_partitioner();
    mpi_comm         = partitioner->get_mpi_communicator();
    local_sums.push_back(local_result_1);
    local_sums.push_back(local_result_2);
    results_queue.push_back(res_1);
    results_queue.push_back(res_2);
  }

  void
  dot_products_commit()
  {
    const std::vector<Number> &     local_sums_tmp = local_sums;
    dealii::ArrayView<const Number> in             = dealii::make_array_view(local_sums_tmp);
    results.resize(local_sums.size());
    dealii::ArrayView<Number> out = dealii::make_array_view(results);
    isum(in, mpi_comm, &mpi_request, out);
  }

  void
  dot_products_finish()
  {
    MPI_Wait(&mpi_request, MPI_STATUS_IGNORE);

    typename std::vector<Number *>::iterator it = results_queue.begin();
    typename std::vector<Number>::iterator   jt = results.begin();
    for (; it != results_queue.end(); ++it, ++jt)
      {
        Number *num = *it;
        (*num)      = *jt;
      }

    local_sums.clear();
    results_queue.clear();
    results.clear();
  };

private:
  // results of local dot products
  std::vector<Number> local_sums;
  // vector of pointers to the values that should be updated
  std::vector<Number *> results_queue;
  // results of allreduce
  std::vector<Number> results;

  MPI_Request mpi_request;
  MPI_Comm    mpi_comm;
};


template <typename VectorType>
class SolverCGPipelined : public dealii::SolverBase<VectorType>
{
public:
  /**
   * Declare type for container size.
   */
  using size_type = dealii::types::global_dof_index;

  /**
   * Constructor.
   */
  SolverCGPipelined(dealii::SolverControl &cn)
    : dealii::SolverBase<VectorType>(cn)
    , times(3, 0.0)
  {}


  /**
   * Virtual destructor.
   */
  virtual ~SolverCGPipelined() override = default;

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
        ScopedTimer timer(times[2]);
        r.equ(-1., b);
      }

    {
      ScopedTimer timer(times[0]);
      r *= -1.0;
      res = r.l2_norm();
    }

    conv = this->iteration_status(0, res, x);
    if (conv != dealii::SolverControl::iterate)
      return;

    {
      ScopedTimer timer(times[1]);
      A.vmult(w, r);
    }

    NonBlockingDotproduct<number, VectorType> nonblocking;

    while (conv == dealii::SolverControl::iterate)
      {
        it++;

        {
          ScopedTimer timer(times[0]);
          nonblocking.dot_product_start(r, r, &gamma);
          nonblocking.dot_product_start(w, r, &delta);
        }
        nonblocking.dot_products_commit();

        {
          ScopedTimer timer(times[1]);
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
        // normal vector updates

        {
          ScopedTimer timer(times[2]);
          z.sadd(beta, 1.0, q);
          s.sadd(beta, 1.0, w);
          p.sadd(beta, 1.0, r);

          x.add(alpha, p);
          r.add(-alpha, s);
          w.add(-alpha, z);
        }

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
