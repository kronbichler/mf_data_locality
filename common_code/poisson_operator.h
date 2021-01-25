
// this file is closely related to the respective file in the
// ceed_benchmarks_dealii repository,
// github.com/kronbichler/ceed_benchmarks_dealii

#ifndef poisson_operator_h
#define poisson_operator_h

#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include "timer.h"
#include "vector_access_reduced.h"

namespace Poisson
{
  using namespace dealii;

  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE Number do_invert(Tensor<2, 2, Number> &t)
  {
    const Number det     = t[0][0] * t[1][1] - t[1][0] * t[0][1];
    const Number inv_det = 1.0 / det;
    const Number tmp     = inv_det * t[0][0];
    t[0][0]              = inv_det * t[1][1];
    t[0][1]              = -inv_det * t[0][1];
    t[1][0]              = -inv_det * t[1][0];
    t[1][1]              = tmp;
    return det;
  }


  template <typename Number>
  inline DEAL_II_ALWAYS_INLINE Number do_invert(Tensor<2, 3, Number> &t)
  {
    const Number tr00    = t[1][1] * t[2][2] - t[1][2] * t[2][1];
    const Number tr10    = t[1][2] * t[2][0] - t[1][0] * t[2][2];
    const Number tr20    = t[1][0] * t[2][1] - t[1][1] * t[2][0];
    const Number det     = t[0][0] * tr00 + t[0][1] * tr10 + t[0][2] * tr20;
    const Number inv_det = 1.0 / det;
    const Number tr01    = t[0][2] * t[2][1] - t[0][1] * t[2][2];
    const Number tr02    = t[0][1] * t[1][2] - t[0][2] * t[1][1];
    const Number tr11    = t[0][0] * t[2][2] - t[0][2] * t[2][0];
    const Number tr12    = t[0][2] * t[1][0] - t[0][0] * t[1][2];
    t[2][1]              = inv_det * (t[0][1] * t[2][0] - t[0][0] * t[2][1]);
    t[2][2]              = inv_det * (t[0][0] * t[1][1] - t[0][1] * t[1][0]);
    t[0][0]              = inv_det * tr00;
    t[0][1]              = inv_det * tr01;
    t[0][2]              = inv_det * tr02;
    t[1][0]              = inv_det * tr10;
    t[1][1]              = inv_det * tr11;
    t[1][2]              = inv_det * tr12;
    t[2][0]              = inv_det * tr20;
    return det;
  }



  template <int dim,
            int fe_degree,
            int n_q_points_1d            = fe_degree + 1,
            int n_components_            = 1,
            typename Number              = double,
            typename VectorType          = LinearAlgebra::distributed::Vector<Number>,
            typename VectorizedArrayType = VectorizedArray<Number>,
            bool use_basic_geometry      = false>
  class LaplaceOperator
  {
  public:
    /**
     * Number typedef.
     */
    typedef Number value_type;

    /**
     * size_type needed for preconditioner classes.
     */
    typedef types::global_dof_index size_type;

    /**
     * Make number of components available as variable
     */
    static constexpr unsigned int n_components = n_components_;

    /**
     * Constructor.
     */
    LaplaceOperator()
    {}

    /**
     * Initialize function.
     */
    void
    initialize(std::shared_ptr<const MatrixFree<dim, Number, VectorizedArrayType>> data_,
               const AffineConstraints<double> &                                   constraints)
    {
      AssertDimension(data_->get_dof_handler().get_fe().n_components(), n_components);
      this->data = data_;
      quad_1d    = QGauss<1>(n_q_points_1d);
      cell_quadratic_coefficients.resize(data->n_cell_batches());

      if (fe_degree > 2)
        {
          compressed_dof_indices.resize(Utilities::pow(3, dim) * VectorizedArrayType::size() *
                                          data->n_cell_batches(),
                                        numbers::invalid_unsigned_int);
          all_indices_uniform.resize(Utilities::pow(3, dim) * data->n_cell_batches(), 1);
        }

      FEValues<dim> fe_values(*data_->get_mapping_info().mapping,
                              data_->get_dof_handler().get_fe(),
                              QGaussLobatto<dim>(3),
                              update_quadrature_points);

      std::vector<types::global_dof_index> dof_indices(
        data->get_dof_handler().get_fe().dofs_per_cell);

      constexpr unsigned int    n_lanes = VectorizedArrayType::size();
      std::vector<unsigned int> renumber_lex =
        FETools::hierarchic_to_lexicographic_numbering<dim>(2);
      for (auto &i : renumber_lex)
        i *= n_lanes;

      for (unsigned int c = 0; c < data->n_cell_batches(); ++c)
        {
          for (unsigned int l = 0; l < data->n_active_entries_per_cell_batch(c); ++l)
            {
              const typename DoFHandler<dim>::cell_iterator cell = data->get_cell_iterator(c, l);
              fe_values.reinit(cell);
              const double           coeff[9] = {1.0, -3.0, 2.0, 0.0, 4.0, -4.0, 0.0, -1.0, 2.0};
              constexpr unsigned int size_dim = Utilities::pow(3, dim);
              std::array<Tensor<1, dim>, size_dim> points;
              for (unsigned int i2 = 0; i2 < (dim > 2 ? 3 : 1); ++i2)
                {
                  for (unsigned int i1 = 0; i1 < 3; ++i1)
                    for (unsigned int i0 = 0, i = 9 * i2 + 3 * i1; i0 < 3; ++i0)
                      points[i + i0] = coeff[i0] * fe_values.quadrature_point(i) +
                                       coeff[i0 + 3] * fe_values.quadrature_point(i + 1) +
                                       coeff[i0 + 6] * fe_values.quadrature_point(i + 2);
                  for (unsigned int i1 = 0; i1 < 3; ++i1)
                    {
                      const unsigned int            i   = 9 * i2 + i1;
                      std::array<Tensor<1, dim>, 3> tmp = {
                        {points[i], points[i + 3], points[i + 6]}};
                      for (unsigned int i0 = 0; i0 < 3; ++i0)
                        points[i + 3 * i0] =
                          coeff[i0] * tmp[0] + coeff[i0 + 3] * tmp[1] + coeff[i0 + 6] * tmp[2];
                    }
                }
              if (dim == 3)
                for (unsigned int i = 0; i < 9; ++i)
                  {
                    std::array<Tensor<1, dim>, 3> tmp = {
                      {points[i], points[i + 9], points[i + 18]}};
                    for (unsigned int i0 = 0; i0 < 3; ++i0)
                      points[i + 9 * i0] =
                        coeff[i0] * tmp[0] + coeff[i0 + 3] * tmp[1] + coeff[i0 + 6] * tmp[2];
                  }
              for (unsigned int i = 0; i < points.size(); ++i)
                for (unsigned int d = 0; d < dim; ++d)
                  cell_quadratic_coefficients[c][i][d][l] = points[i][d];

              if (fe_degree > 2)
                {
                  cell->get_dof_indices(dof_indices);
                  const unsigned int n_components = cell->get_fe().n_components();
                  const unsigned int offset       = Utilities::pow(3, dim) * (n_lanes * c) + l;
                  const Utilities::MPI::Partitioner &part =
                    *data->get_dof_info().vector_partitioner;
                  unsigned int cc = 0, cf = 0;
                  for (unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell;
                       ++i, ++cc, cf += n_components)
                    {
                      if (!constraints.is_constrained(dof_indices[cf]))
                        compressed_dof_indices[offset + renumber_lex[cc]] =
                          part.global_to_local(dof_indices[cf]);
                      for (unsigned int c = 0; c < n_components; ++c)
                        AssertThrow(dof_indices[cf + c] == dof_indices[cf] + c,
                                    ExcMessage("Expected contiguous numbering"));
                    }

                  for (unsigned int line = 0; line < GeometryInfo<dim>::lines_per_cell; ++line)
                    {
                      const unsigned int size = fe_degree - 1;
                      if (!constraints.is_constrained(dof_indices[cf]))
                        {
                          for (unsigned int i = 0; i < size; ++i)
                            for (unsigned int c = 0; c < n_components; ++c)
                              AssertThrow(dof_indices[cf + c * size + i] ==
                                            dof_indices[cf] + i * n_components + c,
                                          ExcMessage("Expected contiguous numbering"));
                          compressed_dof_indices[offset + renumber_lex[cc]] =
                            part.global_to_local(dof_indices[cf]);
                        }
                      ++cc;
                      cf += size * n_components;
                    }
                  for (unsigned int quad = 0; quad < GeometryInfo<dim>::quads_per_cell; ++quad)
                    {
                      const unsigned int size = (fe_degree - 1) * (fe_degree - 1);
                      if (!constraints.is_constrained(dof_indices[cf]))
                        {
                          // switch order x-z for y faces in 3D to lexicographic layout
                          if (dim == 3 && (quad == 2 || quad == 3))
                            for (unsigned int i1 = 0, i = 0; i1 < fe_degree - 1; ++i1)
                              for (unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
                                for (unsigned int c = 0; c < n_components; ++c)
                                  {
                                    AssertThrow(
                                      dof_indices[cf + c * size + i0 * (fe_degree - 1) + i1] ==
                                        dof_indices[cf] + i * n_components + c,
                                      ExcMessage("Expected contiguous numbering"));
                                  }
                          else
                            for (unsigned int i = 0; i < size; ++i)
                              for (unsigned int c = 0; c < n_components; ++c)
                                AssertThrow(dof_indices[cf + c * size + i] ==
                                              dof_indices[cf] + i * n_components + c,
                                            ExcMessage("Expected contiguous numbering"));
                          compressed_dof_indices[offset + renumber_lex[cc]] =
                            part.global_to_local(dof_indices[cf]);
                        }
                      ++cc;
                      cf += size * n_components;
                    }
                  for (unsigned int hex = 0; hex < GeometryInfo<dim>::hexes_per_cell; ++hex)
                    {
                      const unsigned int size = (fe_degree - 1) * (fe_degree - 1) * (fe_degree - 1);
                      if (!constraints.is_constrained(dof_indices[cf]))
                        {
                          for (unsigned int i = 0; i < size; ++i)
                            for (unsigned int c = 0; c < n_components; ++c)
                              AssertThrow(dof_indices[cf + c * size + i] ==
                                            dof_indices[cf] + i * n_components + c,
                                          ExcMessage("Expected contiguous numbering"));
                          compressed_dof_indices[offset + renumber_lex[cc]] =
                            part.global_to_local(dof_indices[cf]);
                        }
                      ++cc;
                      cf += size * n_components;
                    }
                  AssertThrow(cc == Utilities::pow<unsigned int>(3, dim),
                              ExcMessage("Expected 3^dim dofs, got " + std::to_string(cc)));
                  AssertThrow(cf == dof_indices.size(),
                              ExcMessage("Expected (fe_degree+1)^dim dofs, got " +
                                         std::to_string(cf)));
                }
            }
          // insert dummy entries to prevent geometry from degeneration and
          // subsequent division by zero, assuming a Cartesian geometry
          for (unsigned int l = data->n_active_entries_per_cell_batch(c);
               l < VectorizedArrayType::size();
               ++l)
            {
              cell_quadratic_coefficients[c][1][0][l] = 1.;
              if (dim > 1)
                cell_quadratic_coefficients[c][3][1][l] = 1.;
              if (dim > 2)
                cell_quadratic_coefficients[c][9][2][l] = 1.;
            }

          if (fe_degree > 2)
            {
              for (unsigned int i = 0; i < Utilities::pow<unsigned int>(3, dim); ++i)
                for (unsigned int v = 0; v < VectorizedArrayType::size(); ++v)
                  if (compressed_dof_indices[Utilities::pow<unsigned int>(3, dim) *
                                               (VectorizedArrayType::size() * c) +
                                             i * VectorizedArrayType::size() + v] ==
                      numbers::invalid_unsigned_int)
                    all_indices_uniform[Utilities::pow(3, dim) * c + i] = 0;
            }
        }
    }

    /**
     * Initialize function.
     */
    void
    initialize_dof_vector(VectorType &vec) const
    {
      data->initialize_dof_vector(vec);
    }

    /**
     * Matrix-vector multiplication.
     */
    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src, true);
      for (unsigned int i : data->get_constrained_dofs())
        dst.local_element(i) = src.local_element(i);
    }

    /**
     * Matrix-vector multiplication.
     */
    void
    vmult_and_timings(VectorType &      dst,
                      const VectorType &src,
                      double &          time_comp,
                      double &          time_comm) const
    {
      internal::MFWorker<MatrixFree<dim, Number, VectorizedArrayType>,
                         VectorType,
                         VectorType,
                         LaplaceOperator,
                         true>
        funct(*this->data, src, dst, true, *this, &LaplaceOperator::local_apply, nullptr, nullptr);

      {
        ScopedTimer timer(time_comm);
        funct.vector_update_ghosts_start();
        funct.vector_update_ghosts_finish();
      }

      {
        ScopedTimer timer(time_comp);
        const auto &task_info           = this->data->get_task_info();
        const auto &partition_row_index = task_info.partition_row_index;
        const auto &cell_partition_data = task_info.cell_partition_data;

        for (unsigned int part = 0; part < partition_row_index.size() - 2; ++part)
          {
            for (unsigned int i = partition_row_index[part]; i < partition_row_index[part + 1]; ++i)
              {
                funct.zero_dst_vector_range(i);
                AssertIndexRange(i + 1, cell_partition_data.size());
                if (cell_partition_data[i + 1] > cell_partition_data[i])
                  {
                    funct.cell(std::make_pair(cell_partition_data[i], cell_partition_data[i + 1]));
                  }
              }
          }
      }

      {
        ScopedTimer timer(time_comm);
        funct.vector_compress_start();
        funct.vector_compress_finish();
      }

      {
        ScopedTimer timer(time_comp);
        for (unsigned int i : data->get_constrained_dofs())
          dst.local_element(i) = src.local_element(i);
      }
    }

    /**
     * Matrix-vector multiplication.
     */
    void
    vmult(
      VectorType &                                                       dst,
      const VectorType &                                                 src,
      const std::function<void(const unsigned int, const unsigned int)> &operation_before_loop,
      const std::function<void(const unsigned int, const unsigned int)> &operation_after_loop) const
    {
      this->data->cell_loop(
        &LaplaceOperator::local_apply, this, dst, src, operation_before_loop, operation_after_loop);
      for (unsigned int i : data->get_constrained_dofs())
        dst.local_element(i) = src.local_element(i);
    }

    /**
     * Transpose matrix-vector multiplication. Since the Laplace matrix is
     * symmetric, it does exactly the same as vmult().
     */
    void
    Tvmult(VectorType &dst, const VectorType &src) const
    {
      vmult(dst, src);
    }

    /**
     * Compute the diagonal (scalar variant) of the matrix
     */
    LinearAlgebra::distributed::Vector<Number>
    compute_inverse_diagonal() const
    {
      LinearAlgebra::distributed::Vector<Number> diag;
      data->initialize_dof_vector(diag);
      FEEvaluation<dim, fe_degree, n_q_points_1d, 1, Number, VectorizedArrayType> phi(*data);

      AlignedVector<VectorizedArrayType> diagonal(phi.dofs_per_cell);
      for (unsigned int cell = 0; cell < data->n_cell_batches(); ++cell)
        {
          phi.reinit(cell);
          for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
                phi.submit_dof_value(VectorizedArrayType(), j);
              phi.submit_dof_value(make_vectorized_array<VectorizedArrayType>(1.), i);

              phi.evaluate(false, true);
              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                phi.submit_gradient(phi.get_gradient(q), q);
              phi.integrate(false, true);
              diagonal[i] = phi.get_dof_value(i);
            }
          for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
            phi.submit_dof_value(diagonal[i], i);
          phi.distribute_local_to_global(diag);
        }
      diag.compress(VectorOperation::add);
      for (unsigned int i = 0; i < diag.local_size(); ++i)
        if (diag.local_element(i) == 0.)
          diag.local_element(i) = 1.;
        else
          diag.local_element(i) = 1. / diag.local_element(i);
      return diag;
    }

  private:
    void
    local_apply(const MatrixFree<dim, value_type, VectorizedArrayType> &data,
                VectorType &                                            dst,
                const VectorType &                                      src,
                const std::pair<unsigned int, unsigned int> &           cell_range) const
    {
      FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, Number, VectorizedArrayType> phi(
        data);
      constexpr unsigned int n_q_points = Utilities::pow(n_q_points_1d, dim);

      using TensorType = Tensor<1, dim, VectorizedArrayType>;
      std::array<TensorType, Utilities::pow(3, dim - 1)> xi;
      std::array<TensorType, Utilities::pow(3, dim - 1)> di;
      using Eval = dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                            dim,
                                                            n_q_points_1d,
                                                            n_q_points_1d,
                                                            VectorizedArrayType,
                                                            VectorizedArrayType>;

      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
        {
          // Right now, this contains some specialized evaluations with
          // internal data structures that should be generalized/improved and
          // possible come into deal.II at some point
          phi.reinit(cell);
          if (fe_degree > 2)
            read_dof_values_compressed<dim, fe_degree, n_q_points_1d, n_components, value_type>(
              src,
              compressed_dof_indices,
              all_indices_uniform,
              cell,
              phi.get_shape_info().data[0].shape_values_eo,
              phi.get_shape_info().data[0].element_type ==
                dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation,
              phi.begin_values());
          else
            {
              phi.read_dof_values(src);
              phi.evaluate(true, false);
            }
          const auto &         v         = cell_quadratic_coefficients[cell];
          VectorizedArrayType *phi_grads = phi.begin_gradients();
          if (dim == 2)
            {
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval::template apply<1, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi.begin_values() + c * n_q_points,
                    phi_grads + n_q_points + 2 * c * n_q_points);
                  Eval::template apply<0, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi.begin_values(),
                    phi_grads + 2 * c * n_q_points);
                }
              if (use_basic_geometry)
                for (unsigned int q = 0; q < phi.n_q_points; ++q)
                  phi.submit_gradient(phi.get_gradient(q), q);
              else
                for (unsigned int q = 0, qy = 0; qy < n_q_points_1d; ++qy)
                  {
                    const Number     y  = quad_1d.point(qy)[0];
                    const TensorType x1 = v[1] + y * (v[4] + y * v[7]);
                    const TensorType x2 = v[2] + y * (v[5] + y * v[8]);
                    const TensorType d0 = v[3] + (y + y) * v[6];
                    const TensorType d1 = v[4] + (y + y) * v[7];
                    const TensorType d2 = v[5] + (y + y) * v[8];
                    for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
                      {
                        const Number q_weight = quad_1d.weight(qy) * quad_1d.weight(qx);
                        const Number x        = quad_1d.point(qx)[0];
                        Tensor<2, dim, VectorizedArrayType> jac;
                        jac[0]                        = x1 + (x + x) * x2;
                        jac[1]                        = d0 + x * d1 + (x * x) * d2;
                        const VectorizedArrayType det = do_invert(jac);

                        for (unsigned int c = 0; c < n_components; ++c)
                          {
                            const unsigned int  offset = c * dim * n_q_points;
                            VectorizedArrayType tmp[dim];
                            for (unsigned int d = 0; d < dim; ++d)
                              {
                                tmp[d] = jac[d][0] * phi_grads[q + offset];
                                for (unsigned int e = 1; e < dim; ++e)
                                  tmp[d] += jac[d][e] * phi_grads[q + e * n_q_points + offset];
                                tmp[d] *= det * q_weight;
                              }
                            for (unsigned int d = 0; d < dim; ++d)
                              {
                                phi_grads[q + d * n_q_points + offset] = jac[0][d] * tmp[0];
                                for (unsigned int e = 1; e < dim; ++e)
                                  phi_grads[q + d * n_q_points + offset] += jac[e][d] * tmp[e];
                              }
                          }
                      }
                  }
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval::template apply<0, false, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + 2 * c * n_q_points,
                    phi.begin_values());
                  Eval::template apply<1, false, true, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + n_q_points + 2 * c * n_q_points,
                    phi.begin_values() + c * n_q_points);
                }
            }
          else if (dim == 3)
            {
              constexpr unsigned int n_q_points_2d = n_q_points_1d * n_q_points_1d;
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  if (fe_degree > 2 &&
                      phi.get_shape_info().data[0].element_type !=
                        dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation)
                    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                             dim,
                                                             fe_degree + 1,
                                                             n_q_points_1d,
                                                             VectorizedArrayType,
                                                             VectorizedArrayType>::
                      template apply<2, true, false, 0>(
                        phi.get_shape_info().data[0].shape_values_eo.begin(),
                        phi.begin_values() + c * n_q_points,
                        phi.begin_values() + c * n_q_points);
                  Eval::template apply<2, true, false, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi.begin_values() + c * n_q_points,
                    phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points);
                }
              for (unsigned int q = 0, qz = 0; qz < n_q_points_1d; ++qz)
                {
                  using Eval2 =
                    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                             2,
                                                             n_q_points_1d,
                                                             n_q_points_1d,
                                                             VectorizedArrayType,
                                                             VectorizedArrayType>;
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      Eval2::template apply<1, true, false, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d,
                        phi_grads + (2 * c + 1) * n_q_points_2d);
                      Eval2::template apply<0, true, false, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d,
                        phi_grads + 2 * c * n_q_points_2d);
                    }
                  if (use_basic_geometry)
                    {
                      const Tensor<2, dim, VectorizedArrayType> *jac =
                        data.get_mapping_info().cell_data[0].jacobians[0].begin() +
                        data.get_mapping_info().cell_data[0].data_index_offsets[cell];
                      const VectorizedArrayType *jxw =
                        data.get_mapping_info().cell_data[0].JxW_values.begin() +
                        data.get_mapping_info().cell_data[0].data_index_offsets[cell];
                      const Number *weights = data.get_mapping_info()
                                                .cell_data[0]
                                                .descriptor[0]
                                                .quadrature_weights.begin();
                      if (data.get_mapping_info().cell_type[cell] <=
                          internal::MatrixFreeFunctions::affine)
                        for (unsigned int q1 = 0; q1 < n_q_points_2d; ++q1, ++q)
                          {
                            const VectorizedArrayType det = jxw[0] * weights[q];
                            for (unsigned int c = 0; c < n_components; ++c)
                              {
                                VectorizedArrayType tmp[dim], tmp2[dim];
                                for (unsigned int d = 0; d < dim; ++d)
                                  {
                                    tmp[d] =
                                      jac[0][d][0] * phi_grads[q1 + c * 2 * n_q_points_2d] +
                                      jac[0][d][1] * phi_grads[q1 + (c * 2 + 1) * n_q_points_2d] +
                                      jac[0][d][2] *
                                        phi_grads[q + 2 * n_components * n_q_points_2d +
                                                  c * n_q_points];
                                    tmp[d] *= det;
                                  }
                                for (unsigned int d = 0; d < dim; ++d)
                                  {
                                    tmp2[d] = jac[0][0][d] * tmp[0];
                                    for (unsigned int e = 1; e < dim; ++e)
                                      tmp2[d] += jac[0][e][d] * tmp[e];
                                  }
                                phi_grads[q1 + c * 2 * n_q_points_2d]       = tmp2[0];
                                phi_grads[q1 + (c * 2 + 1) * n_q_points_2d] = tmp2[1];
                                phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points] =
                                  tmp2[2];
                              }
                          }
                      else
                        for (unsigned int q1 = 0; q1 < n_q_points_2d; ++q1, ++q)
                          for (unsigned int c = 0; c < n_components; ++c)
                            {
                              VectorizedArrayType tmp[dim], tmp2[dim];
                              for (unsigned int d = 0; d < dim; ++d)
                                {
                                  tmp[d] =
                                    jac[q][d][0] * phi_grads[q1 + c * 2 * n_q_points_2d] +
                                    jac[q][d][1] * phi_grads[q1 + (c * 2 + 1) * n_q_points_2d] +
                                    jac[q][d][2] * phi_grads[q + 2 * n_components * n_q_points_2d +
                                                             c * n_q_points];
                                  tmp[d] *= jxw[q];
                                }
                              for (unsigned int d = 0; d < dim; ++d)
                                {
                                  tmp2[d] = jac[q][0][d] * tmp[0];
                                  for (unsigned int e = 1; e < dim; ++e)
                                    tmp2[d] += jac[q][e][d] * tmp[e];
                                }
                              phi_grads[q1 + c * 2 * n_q_points_2d]       = tmp2[0];
                              phi_grads[q1 + (c * 2 + 1) * n_q_points_2d] = tmp2[1];
                              phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points] =
                                tmp2[2];
                            }
                    }
                  else
                    {
                      const Number z = quad_1d.point(qz)[0];
                      di[0]          = v[9] + (z + z) * v[18];
                      for (unsigned int i = 1; i < 9; ++i)
                        {
                          xi[i] = v[i] + z * (v[9 + i] + z * v[18 + i]);
                          di[i] = v[9 + i] + (z + z) * v[18 + i];
                        }
                      for (unsigned int qy = 0; qy < n_q_points_1d; ++qy)
                        {
                          const auto       y            = quad_1d.point(qy)[0];
                          const TensorType x1           = xi[1] + y * (xi[4] + y * xi[7]);
                          const TensorType x2           = xi[2] + y * (xi[5] + y * xi[8]);
                          const TensorType dy0          = xi[3] + (y + y) * xi[6];
                          const TensorType dy1          = xi[4] + (y + y) * xi[7];
                          const TensorType dy2          = xi[5] + (y + y) * xi[8];
                          const TensorType dz0          = di[0] + y * (di[3] + y * di[6]);
                          const TensorType dz1          = di[1] + y * (di[4] + y * di[7]);
                          const TensorType dz2          = di[2] + y * (di[5] + y * di[8]);
                          double           q_weight_tmp = quad_1d.weight(qz) * quad_1d.weight(qy);
                          for (unsigned int qx = 0; qx < n_q_points_1d; ++qx, ++q)
                            {
                              const Number                        x = quad_1d.point(qx)[0];
                              Tensor<2, dim, VectorizedArrayType> jac;
                              jac[0]                  = x1 + (x + x) * x2;
                              jac[1]                  = dy0 + x * (dy1 + x * dy2);
                              jac[2]                  = dz0 + x * (dz1 + x * dz2);
                              VectorizedArrayType det = do_invert(jac);
                              det                     = det * (q_weight_tmp * quad_1d.weight(qx));

                              for (unsigned int c = 0; c < n_components; ++c)
                                {
                                  VectorizedArrayType tmp[dim], tmp2[dim];
                                  for (unsigned int d = 0; d < dim; ++d)
                                    {
                                      tmp[d] =
                                        jac[d][0] * phi_grads[qy * n_q_points_1d + qx +
                                                              c * 2 * n_q_points_2d] +
                                        jac[d][1] * phi_grads[qy * n_q_points_1d + qx +
                                                              (c * 2 + 1) * n_q_points_2d] +
                                        jac[d][2] * phi_grads[q + 2 * n_components * n_q_points_2d +
                                                              c * n_q_points];
                                      tmp[d] *= det;
                                    }
                                  for (unsigned int d = 0; d < dim; ++d)
                                    {
                                      tmp2[d] = jac[0][d] * tmp[0];
                                      for (unsigned int e = 1; e < dim; ++e)
                                        tmp2[d] += jac[e][d] * tmp[e];
                                    }
                                  phi_grads[qy * n_q_points_1d + qx + c * 2 * n_q_points_2d] =
                                    tmp2[0];
                                  phi_grads[qy * n_q_points_1d + qx + (c * 2 + 1) * n_q_points_2d] =
                                    tmp2[1];
                                  phi_grads[q + 2 * n_components * n_q_points_2d + c * n_q_points] =
                                    tmp2[2];
                                }
                            }
                        }
                    }
                  for (unsigned int c = 0; c < n_components; ++c)
                    {
                      Eval2::template apply<0, false, false, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi_grads + 2 * c * n_q_points_2d,
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d);
                      Eval2::template apply<1, false, true, 1>(
                        phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                        phi_grads + (2 * c + 1) * n_q_points_2d,
                        phi.begin_values() + c * n_q_points + qz * n_q_points_2d);
                    }
                }
              for (unsigned int c = 0; c < n_components; ++c)
                {
                  Eval::template apply<2, false, true, 1>(
                    phi.get_shape_info().data[0].shape_gradients_collocation_eo.begin(),
                    phi_grads + (2 * n_components * n_q_points_2d) + c * n_q_points,
                    phi.begin_values() + c * n_q_points);
                  if (fe_degree > 2 &&
                      phi.get_shape_info().data[0].element_type !=
                        dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation)
                    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_evenodd,
                                                             dim,
                                                             fe_degree + 1,
                                                             n_q_points_1d,
                                                             VectorizedArrayType,
                                                             VectorizedArrayType>::
                      template apply<2, false, false, 0>(
                        phi.get_shape_info().data[0].shape_values_eo.begin(),
                        phi.begin_values() + c * n_q_points,
                        phi.begin_values() + c * n_q_points);
                }
            }

          if (fe_degree > 2)
            distribute_local_to_global_compressed<dim,
                                                  fe_degree,
                                                  n_q_points_1d,
                                                  n_components,
                                                  Number>(
              dst,
              compressed_dof_indices,
              all_indices_uniform,
              cell,
              phi.get_shape_info().data[0].shape_values_eo,
              phi.get_shape_info().data[0].element_type ==
                dealii::internal::MatrixFreeFunctions::tensor_symmetric_collocation,
              phi.begin_values());
          else
            phi.integrate_scatter(true, false, dst);
        }
    }

    std::shared_ptr<const MatrixFree<dim, Number, VectorizedArrayType>> data;

    Quadrature<1> quad_1d;
    AlignedVector<std::array<Tensor<1, dim, VectorizedArrayType>, Utilities::pow(3, dim)>>
      cell_quadratic_coefficients;

    std::vector<unsigned int>  compressed_dof_indices;
    std::vector<unsigned char> all_indices_uniform;
  };
} // namespace Poisson

#endif
