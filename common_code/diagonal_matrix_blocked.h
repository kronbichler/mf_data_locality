
#ifndef diagonal_matrix_blocked_h
#define diagonal_matrix_blocked_h

#include <deal.II/lac/la_parallel_vector.h>


template <int dim, typename Number>
class DiagonalMatrixBlocked
{
public:
  void
  vmult(dealii::LinearAlgebra::distributed::Vector<Number> &      dst,
        const dealii::LinearAlgebra::distributed::Vector<Number> &src) const
  {
    AssertThrow(dst.size() == dim * diagonal.size(),
                dealii::ExcNotImplemented("Dimension mismatch " + std::to_string(dst.size()) +
                                          " vs " + std::to_string(dim) + " x " +
                                          std::to_string(diagonal.size())));
    if (dim == 1)
      {
        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int i = 0; i < diagonal.locally_owned_size(); ++i)
          dst.local_element(i) = diagonal.local_element(i) * src.local_element(i);
      }
    else
      for (unsigned int i = 0, c = 0; i < diagonal.locally_owned_size(); ++i)
        {
          const Number diag = diagonal.local_element(i);
          for (unsigned int d = 0; d < dim; ++d, ++c)
            dst.local_element(c) = diag * src.local_element(c);
        }
  }

  const dealii::LinearAlgebra::distributed::Vector<Number> &
  get_vector() const
  {
    return diagonal;
  }

  dealii::LinearAlgebra::distributed::Vector<Number> diagonal;
};

#endif
