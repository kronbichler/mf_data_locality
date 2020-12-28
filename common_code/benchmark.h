
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
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
#include "renumber_dofs_for_mf.h"

using namespace dealii;

#define USE_SHMEM

// Define the number of components in the benchmark
constexpr unsigned int dimension    = 3;
constexpr unsigned int n_components = dimension;


template <typename Operator, typename Preconditioner>
std::pair<unsigned int, std::vector<double>>
run_cg_solver(const Operator &                                  laplace_operator,
              LinearAlgebra::distributed::Vector<double> &      x,
              const LinearAlgebra::distributed::Vector<double> &b,
              const Preconditioner &                            preconditioner);


template <int dim, int fe_degree, int n_q_points>
void
run_templated(const unsigned int s, const bool short_output, const MPI_Comm &comm_shmem)
{
#ifndef USE_SHMEM
  (void)comm_shmem;
#endif

  warmup_code();

  if (short_output == true)
    deallog.depth_console(0);
  else if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog.depth_console(2);

  Timer              time;
  unsigned int       n_refine  = s / 6;
  const unsigned int remainder = s % 6;

  MyManifold<dim>                           manifold;
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  std::vector<unsigned int>                 subdivisions(dim, 1);
  if (remainder == 1 && s > 1)
    {
      subdivisions[0] = 3;
      subdivisions[1] = 2;
      subdivisions[2] = 2;
      n_refine -= 1;
    }
  if (remainder == 2)
    subdivisions[0] = 2;
  else if (remainder == 3)
    subdivisions[0] = 3;
  else if (remainder == 4)
    subdivisions[0] = subdivisions[1] = 2;
  else if (remainder == 5)
    {
      subdivisions[0] = 3;
      subdivisions[1] = 2;
    }

  Point<dim> p2;
  for (unsigned int d = 0; d < dim; ++d)
    p2[d] = subdivisions[d];
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
  mf_data.communicator_sm                = comm_shmem;
  mf_data.use_vector_data_exchanger_full = true;
#endif

  // renumber Dofs to minimize the number of partitions in import indices of
  // partitioner
  Renumber<dim, double> renum(0, 1, 2);
  renum.renumber(dof_handler, constraints, mf_data);

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
  if (short_output == false)
    {
      const double diag_norm = diag_mat.diagonal.l2_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Norm of diagonal for preconditioner: " << diag_norm << std::endl;
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

  // TODO: we want to fill in a proper right hand side that allows us to
  // compute a manufactured solution
  LinearAlgebra::distributed::Vector<double> input, output, tmp;
  laplace_operator.initialize_dof_vector(input);
  laplace_operator.initialize_dof_vector(output);
  laplace_operator.initialize_dof_vector(tmp);
  for (unsigned int i = 0; i < input.local_size(); ++i)
    if (!constraints.is_constrained(input.get_partitioner()->local_to_global(i)))
      input.local_element(i) = i % 8;

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && short_output == false)
    std::cout << "Setup time:         " << data.min << " (p" << data.min_index << ") " << data.avg
              << " " << data.max << " (p" << data.max_index << ")"
              << "s" << std::endl;

  double              solver_time  = 1e10;
  unsigned int        n_iterations = numbers::invalid_unsigned_int;
  std::vector<double> profile;
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("cg_solver");
#endif
  for (unsigned int t = 0; t < 4; ++t)
    {
      output = 0;
      time.restart();
      const auto temp = run_cg_solver(laplace_operator, output, input, diag_mat);
      n_iterations    = temp.first;
      data            = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);

      if (data.max < solver_time)
        {
          solver_time = data.max;
          profile     = temp.second;
        }
    }
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("cg_solver");
#endif

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START("matvec");
#endif
  double matvec_time = 1e10;
  for (unsigned int t = 0; t < 2; ++t)
    {
      time.restart();
      for (unsigned int i = 0; i < 50; ++i)
        laplace_operator.vmult(input, output);
      data        = Utilities::MPI::min_max_avg(time.wall_time(), MPI_COMM_WORLD);
      matvec_time = std::min(data.max / 50, matvec_time);
    }
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP("matvec");
#endif

  if (short_output == false)
    {
      matrix_free->template cell_loop<LinearAlgebra::distributed::Vector<double>,
                                      LinearAlgebra::distributed::Vector<double>>(
        [](const auto &data, auto &dst, const auto &src, const auto &range) {
          FEEvaluation<dim, fe_degree, n_q_points, n_components, double> eval(data);
          for (unsigned int cell = range.first; cell < range.second; ++cell)
            {
              eval.reinit(cell);
              eval.gather_evaluate(src, false, true);
              for (unsigned int q = 0; q < eval.n_q_points; ++q)
                eval.submit_gradient(eval.get_gradient(q), q);
              eval.integrate_scatter(false, true, dst);
            }
        },
        tmp,
        output,
        true);
      tmp -= input;
      const double error = tmp.l2_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Error mat-vec:         " << error << std::endl;
    }

#if false
  const auto profile_min_max_avg = Utilities::MPI::min_max_avg(profile, MPI_COMM_WORLD);
#endif

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && short_output == true)
    {
      std::cout << std::setprecision(4)                                                      //
                << std::setw(2) << fe_degree << " " << std::setw(2) << n_q_points            //
                << " " << std::setw(10) << tria.n_global_active_cells()                      //
                << " " << std::setw(11) << dof_handler.n_dofs()                              //
                << " " << std::setw(9) << solver_time / n_iterations                         //
                << " " << std::setw(10) << dof_handler.n_dofs() / solver_time * n_iterations //
                << " " << std::setw(4) << n_iterations                                       //
                << " " << std::setw(9) << matvec_time;                                       //

#if false
      for (const auto &t : profile_min_max_avg)
        std::cout << " " << std::setw(9) << t.avg / std::max(1u, n_iterations);
#else
      for (const auto &t : profile)
        std::cout << " " << std::setw(9) << t / std::max(1u, n_iterations);
#endif

      std::cout << std::endl;
    }
}


template <int dim, int fe_degree, int n_q_points>
void
do_test(const int s_in, const bool compact_output)
{
  MPI_Comm comm_shmem;

#ifdef USE_SHMEM
  MPI_Comm_split_type(MPI_COMM_WORLD,
                      MPI_COMM_TYPE_SHARED,
                      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD),
                      MPI_INFO_NULL,
                      &comm_shmem);
#endif

  if (s_in < 1)
    {
      unsigned int s = 1 + std::log2((Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) + 1) / 2);
      // std::max(3U, static_cast<unsigned int>
      //         (std::log2(1024/fe_degree/fe_degree/fe_degree)));
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << " p  q      cells        dofs  timeCGit throughput itCG    timeMV"
                  << std::endl;
      while (Utilities::fixed_power<dim>(fe_degree + 1) * (1UL << (s / 2)) * n_components <
             6000000ULL * Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        {
          run_templated<dim, fe_degree, n_q_points>(s, compact_output, comm_shmem);
          ++s;
        }
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << std::endl << std::endl;
    }
  else
    run_templated<dim, fe_degree, n_q_points>(2 * s_in, compact_output, comm_shmem);

#ifdef USE_SHMEM
  MPI_Comm_free(&comm_shmem);
#endif
}


void
run(int argc, char **argv)
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  unsigned int degree         = 1;
  unsigned int s              = -1;
  bool         compact_output = true;
  if (argc > 1)
    degree = std::atoi(argv[1]);
  if (argc > 2)
    s = std::atoi(argv[2]);
  if (argc > 3)
    compact_output = std::atoi(argv[3]);

  if (degree == 1)
    do_test<dimension, 1, 3>(s, compact_output);
  else if (degree == 2)
    do_test<dimension, 2, 4>(s, compact_output);
  else if (degree == 3)
    do_test<dimension, 3, 5>(s, compact_output);
  else if (degree == 4)
    do_test<dimension, 4, 6>(s, compact_output);
  else if (degree == 5)
    do_test<dimension, 5, 7>(s, compact_output);
  else if (degree == 6)
    do_test<dimension, 6, 8>(s, compact_output);
  else if (degree == 7)
    do_test<dimension, 7, 9>(s, compact_output);
  else if (degree == 8)
    do_test<dimension, 8, 10>(s, compact_output);
  else if (degree == 9)
    do_test<dimension, 9, 11>(s, compact_output);
  else if (degree == 10)
    do_test<dimension, 10, 12>(s, compact_output);
  else if (degree == 11)
    do_test<dimension, 11, 13>(s, compact_output);
  else
    AssertThrow(false, ExcMessage("Only degrees up to 11 implemented"));

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
