
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/vector_tools.h>

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

#include "curved_manifold.h"
#include "diagonal_matrix_blocked.h"
#include "poisson_operator.h"

using namespace dealii;

//#define USE_SHMEM

// Define the number of components in the benchmark
constexpr unsigned int dimension    = 3;
constexpr unsigned int n_components = dimension;


template <typename Operator, typename Preconditioner>
unsigned int
run_cg_solver(const Operator &                                  laplace_operator,
              LinearAlgebra::distributed::Vector<double> &      x,
              const LinearAlgebra::distributed::Vector<double> &b,
              const Preconditioner &                            preconditioner);



template <int dim>
class AnalyticalSolution : public Function<dim>
{
public:
  AnalyticalSolution()
    : Function<dim>(dim)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component) const override
  {
    double value = 1;
    for (unsigned int d = 0; d < dim; ++d)
      value *= std::sin(numbers::PI * (d + component + 1) * p[d]);
    return value;
  }

  double
  laplacian(const Point<dim> &p, const unsigned int component) const override
  {
    double factor = 0;
    for (unsigned int d = 0; d < dim; ++d)
      factor -= Utilities::fixed_power<2>(numbers::PI * (d + component + 1));
    return factor * value(p, component);
  }
};



template <int dim, int fe_degree, int n_q_points>
void
run_templated(const unsigned int s, const MPI_Comm &comm_shmem, ConvergenceTable &table)
{
#ifndef USE_SHMEM
  (void)comm_shmem;
#endif

  const unsigned int n_refine  = s / 3;
  const unsigned int remainder = s % 3;
  Point<dim>         p2;
  for (unsigned int d = 0; d < remainder; ++d)
    p2[d] = 2;
  for (unsigned int d = remainder; d < dim; ++d)
    p2[d] = 1;

  MyManifold<dim>                           manifold;
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  std::vector<unsigned int>                 subdivisions(dim, 1);
  for (unsigned int d = 0; d < remainder; ++d)
    subdivisions[d] = 2;
  GridGenerator::subdivided_hyper_rectangle(tria, subdivisions, Point<dim>(), p2);
  GridTools::transform(std::bind(&MyManifold<dim>::push_forward, manifold, std::placeholders::_1),
                       tria);
  tria.set_all_manifold_ids(1);
  tria.set_manifold(1, manifold);
  tria.refine_global(n_refine);
  MappingQGeneric<dim> mapping(2);

  FE_Q<dim>       fe_scalar(fe_degree);
  FESystem<dim>   fe_q(fe_scalar, n_components);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe_q);

  AffineConstraints<double> constraints;
  IndexSet                  relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(n_components), constraints);
  constraints.close();
  typename MatrixFree<dim, double>::AdditionalData mf_data;

#ifdef USE_SHMEM
  mf_data.communicator_sm = comm_shmem;
#endif
  mf_data.mapping_update_flags |= update_quadrature_points;

  // renumber Dofs to minimize the number of partitions in import indices of
  // partitioner
  DoFRenumbering::matrix_free_data_locality(dof_handler, constraints, mf_data);

  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.clear();
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(n_components), constraints);
  constraints.close();

  std::shared_ptr<MatrixFree<dim, double>> matrix_free(new MatrixFree<dim, double>());

  // create preconditioner based on the diagonal of the GLL quadrature with
  // fe_degree+1 points
  DiagonalMatrixBlocked<n_components, double> diag_mat;
  {
    matrix_free->reinit(
      mapping, dof_handler, constraints, QGaussLobatto<1>(fe_degree + 1), mf_data);

    Poisson::LaplaceOperator<dim,
                             fe_degree,
                             fe_degree + 1,
                             n_components,
                             double,
                             LinearAlgebra::distributed::Vector<double>>
      laplace_operator;
    laplace_operator.initialize(matrix_free, constraints);

    const auto vector = laplace_operator.compute_inverse_diagonal();
    IndexSet   reduced(vector.size() / n_components);
    reduced.add_range(vector.get_partitioner()->local_range().first / n_components,
                      vector.get_partitioner()->local_range().second / n_components);
    reduced.compress();
    diag_mat.diagonal.reinit(reduced, vector.get_mpi_communicator());
    for (unsigned int i = 0; i < reduced.n_elements(); ++i)
      diag_mat.diagonal.local_element(i) = vector.local_element(i * n_components);
  }

  // now to the actual operator with the given number of quadrature points
  matrix_free->reinit(mapping, dof_handler, constraints, QGauss<1>(n_q_points), mf_data);

  Poisson::LaplaceOperator<dim,
                           fe_degree,
                           n_q_points,
                           n_components,
                           double,
                           LinearAlgebra::distributed::Vector<double>>
    laplace_operator;
  laplace_operator.initialize(matrix_free, constraints);

  LinearAlgebra::distributed::Vector<double> input, output;
  laplace_operator.initialize_dof_vector(input);
  laplace_operator.initialize_dof_vector(output);

  matrix_free->template cell_loop<LinearAlgebra::distributed::Vector<double>,
                                  LinearAlgebra::distributed::Vector<double>>(
    [](const auto &data, auto &dst, const auto &, const auto &range) {
      FEEvaluation<dim, fe_degree, n_q_points, n_components, double> eval(data);
      for (unsigned int cell = range.first; cell < range.second; ++cell)
        {
          eval.reinit(cell);
          for (unsigned int q = 0; q < eval.n_q_points; ++q)
            {
              const auto                              p_vec = eval.quadrature_point(q);
              Tensor<1, dim, VectorizedArray<double>> laplacian;
              for (unsigned int v = 0; v < laplacian[0].size(); ++v)
                {
                  Point<dim> p;
                  for (unsigned int d = 0; d < dim; ++d)
                    p[d] = p_vec[d][v];
                  AnalyticalSolution<dim> solution;
                  for (unsigned int d = 0; d < dim; ++d)
                    laplacian[d][v] = solution.laplacian(p, d);
                }
              eval.submit_value(-laplacian, q);
            }
          eval.integrate_scatter(true, false, dst);
        }
    },
    input,
    output,
    true);

  const unsigned int n_iterations = run_cg_solver(laplace_operator, output, input, diag_mat);
  output.update_ghost_values();
  Vector<double> cellwise_error(tria.n_active_cells());
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    output,
                                    AnalyticalSolution<dim>(),
                                    cellwise_error,
                                    QGauss<dim>(fe_degree + 2),
                                    VectorTools::L2_norm);

  table.add_value("dim", dim);
  table.add_value("degree", fe_degree);
  table.add_value("s", s);
  table.add_value("n_cells", tria.n_global_active_cells());
  table.add_value("n_dofs", dof_handler.n_dofs());
  table.add_value("n_it", n_iterations >= 100);
  table.add_value("error",
                  VectorTools::compute_global_error(tria, cellwise_error, VectorTools::L2_norm) /
                    std::sqrt(1 << (s % dim)));
  table.set_scientific("error", true);
}


template <int dim, int fe_degree, int n_q_points>
void
do_test(ConvergenceTable &table, const unsigned int min_s)
{
  MPI_Comm comm_shmem;

#ifdef USE_SHMEM
  MPI_Comm_split_type(MPI_COMM_WORLD,
                      MPI_COMM_TYPE_SHARED,
                      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
                      MPI_INFO_NULL,
                      &comm_shmem);
#else
  comm_shmem = MPI_COMM_SELF;
#endif

  unsigned int s = std::max<unsigned int>(min_s,
                                          1 + static_cast<unsigned int>(std::log2(
                                                Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))));
  while (Utilities::fixed_power<dim>(fe_degree + 1) * (1UL << s) * n_components <
         60000ULL * Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    {
      run_templated<dim, fe_degree, n_q_points>(s, comm_shmem, table);
      ++s;
    }

#ifdef USE_SHMEM
  MPI_Comm_free(&comm_shmem);
#endif
}


void
run(const unsigned int min_s = 1)
{
  ConvergenceTable table;

  do_test<dimension, 1, 3>(table, min_s);
  do_test<dimension, 2, 4>(table, min_s);
  do_test<dimension, 3, 5>(table, min_s);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    table.write_text(std::cout);
}
