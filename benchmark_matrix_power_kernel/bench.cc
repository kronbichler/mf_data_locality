
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_renumbering.h>

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

#include "../common_code/curved_manifold.h"
#include "../common_code/diagonal_matrix_blocked.h"
#include "../common_code/poisson_operator.h"
#include "../common_code/renumber_dofs_for_mf.h"

using namespace dealii;

template <int dim>
void
run(const unsigned int s, const unsigned int fe_degree, const unsigned int n_components = 1)
{
  using Number              = double;
  using VectorizedArrayType = VectorizedArray<Number>;

  unsigned int       n_refine  = s / 6;
  const unsigned int remainder = s % 6;

  Triangulation<dim>        tria;
  std::vector<unsigned int> subdivisions(dim, 1);
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

  mf_data.mapping_update_flags = update_gradients;
  mf_data.tasks_parallel_scheme =
    MatrixFree<dim, double, VectorizedArrayType>::AdditionalData::TasksParallelScheme::none;

  Renumber<dim, double> renum(0, 1, 2);
  renum.renumber(dof_handler, constraints, mf_data);

  DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  constraints.clear();
  constraints.reinit(relevant_dofs);
  VectorTools::interpolate_boundary_values(
    mapping, dof_handler, 0, Functions::ZeroFunction<dim>(n_components), constraints);
  constraints.close();

  MatrixFree<dim, double> matrix_free;

  matrix_free.reinit(mapping, dof_handler, constraints, QGaussLobatto<1>(fe_degree + 1), mf_data);


  std::vector<unsigned int> min_vector(matrix_free.n_cell_batches() * VectorizedArrayType::size(),
                                       numbers::invalid_unsigned_int);
  std::vector<unsigned int> max_vector(matrix_free.n_cell_batches() * VectorizedArrayType::size(),
                                       numbers::invalid_unsigned_int);


  std::vector<std::vector<unsigned int>> vertices_to_cells(tria.n_vertices());

  double dummy = 0;
  matrix_free.template loop_cell_centric<double, double>(
    [&](const auto &data, auto &, const auto &, const auto cells) {
      (void)data;

      for (unsigned int cell = cells.first; cell < cells.second; ++cell)
        {
          const auto n_active_entries_per_cell_batch = data.n_active_entries_per_cell_batch(cell);

          for (unsigned int v = 0; v < n_active_entries_per_cell_batch; ++v)
            {
              const auto cell_iterator = matrix_free.get_cell_iterator(cell, v);

              for (const auto i : cell_iterator->vertex_indices())
                vertices_to_cells[cell_iterator->vertex_index(i)].push_back(
                  cell * VectorizedArrayType::size() + v);
            }
        }
    },
    dummy,
    dummy);

  unsigned int counter = 0;
  matrix_free.template loop_cell_centric<double, double>(
    [&](const auto &data, auto &, const auto &, const auto cells) {
      (void)data;

      for (unsigned int cell = cells.first; cell < cells.second; ++cell)
        {
          const auto n_active_entries_per_cell_batch = data.n_active_entries_per_cell_batch(cell);


          // std::cout << n_active_entries_per_cell_batch << std::endl;

          for (unsigned int v = 0; v < n_active_entries_per_cell_batch; ++v)
            {
              const auto cell_iterator = matrix_free.get_cell_iterator(cell, v);

              for (const auto i : cell_iterator->vertex_indices())
                for (const auto cell_index : vertices_to_cells[cell_iterator->vertex_index(i)])
                  {
                    if (min_vector[cell_index] == numbers::invalid_unsigned_int)
                      min_vector[cell * VectorizedArrayType::size() + v] = counter;

                    max_vector[cell_index] = counter;
                  }
            }
        }
      counter++;

      // std::cout << cells.first << " " << cells.second << std::endl;
    },
    dummy,
    dummy);

  const auto process = [counter](const auto &ids) {
    std::vector<std::vector<unsigned int>> temp(counter);

    for (unsigned int i = 0; i < ids.size(); ++i)
      if (ids[i] != numbers::invalid_unsigned_int)
        temp[ids[i]].push_back(i);

    return temp;
  };

  const auto pre_indices  = process(min_vector);
  const auto post_indices = process(max_vector);

  using VectorType = LinearAlgebra::distributed::Vector<double>;
  VectorType src, dst_0, dst_1, dst_2;

  matrix_free.initialize_dof_vector(src);
  matrix_free.initialize_dof_vector(dst_0);
  matrix_free.initialize_dof_vector(dst_1);
  matrix_free.initialize_dof_vector(dst_2);


  const auto process_batch = [](const auto &id, auto &phi, auto &dst, const auto &src) {
    phi.reinit(id);
    phi.read_dof_values(src);
    phi.evaluate(EvaluationFlags::gradients);

    for (const auto q : phi.quadrature_point_indices())
      phi.submit_gradient(phi.get_gradient(q), q);

    phi.integrate(EvaluationFlags::gradients);
    phi.distribute_local_to_global(dst);
  };

  MPI_Barrier(MPI_COMM_WORLD);

  auto temp_time = std::chrono::system_clock::now();

  for (unsigned int c = 0; c < 10; ++c)
    {
      counter = 0;
      matrix_free.template loop_cell_centric<double, double>(
        [&](const auto &data, auto &, const auto &, const auto cells) {
          FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi(data);
          FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi_(data);

          // pre vmult
          if (true)
            for (unsigned int i = 0; i < pre_indices[counter].size();
                 i += VectorizedArrayType::size())
              {
                std::array<unsigned int, VectorizedArrayType::size()> ids = {};
                ids.fill(numbers::invalid_unsigned_int);

                for (unsigned int v = 0;
                     v < std::min(VectorizedArrayType::size(), pre_indices[counter].size() - i);
                     ++v)
                  ids[v] = pre_indices[counter][i + v];

                process_batch(ids, phi, dst_0, src);
              }

          // vmult
          if (true)
            for (unsigned int cell = cells.first; cell < cells.second; ++cell)
              {
                process_batch(cell, phi_, dst_1, dst_0);
              }

          // post vmult
          if (true)
            for (unsigned int i = 0; i < post_indices[counter].size();
                 i += VectorizedArrayType::size())
              {
                std::array<unsigned int, VectorizedArrayType::size()> ids = {};
                ids.fill(numbers::invalid_unsigned_int);

                for (unsigned int v = 0;
                     v < std::min(VectorizedArrayType::size(), post_indices[counter].size() - i);
                     ++v)
                  ids[v] = post_indices[counter][i + v];

                process_batch(ids, phi, dst_2, dst_1);
              }

          counter++;
        },
        dummy,
        dummy);
    }

  MPI_Barrier(MPI_COMM_WORLD);

  const double time_power = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::system_clock::now() - temp_time)
                              .count() /
                            1e9;


  MPI_Barrier(MPI_COMM_WORLD);

  temp_time = std::chrono::system_clock::now();
  for (unsigned int c = 0; c < 10; ++c)
    matrix_free.template loop_cell_centric<double, double>(
      [&](const auto &data, auto &, const auto &, const auto cells) {
        FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi(data);

        for (unsigned int cell = cells.first; cell < cells.second; ++cell)
          process_batch(cell, phi, dst_1, dst_0);
      },
      dummy,
      dummy);

  MPI_Barrier(MPI_COMM_WORLD);

  const double time_normal = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::system_clock::now() - temp_time)
                               .count() /
                             1e9;


  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << tria.n_active_cells() << " " << dof_handler.n_dofs() << " "
              << (time_power / time_normal) << " " << time_power << " " << time_normal << std::endl;
}



int
main(int argc, char **argv)
{
  // mpirun -np 40 ./benchmark_matrix_power_kernel/bench 3 5 34
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  AssertThrow(argc > 3, ExcNotImplemented());

  const unsigned int dim     = std::atoi(argv[1]);
  const unsigned int degree  = std::atoi(argv[2]);
  const unsigned int n_steps = std::atoi(argv[3]);

  if (dim == 2)
    run<3>(n_steps, degree);
  else if (dim == 3)
    run<3>(n_steps, degree);
  else
    AssertThrow(false, ExcNotImplemented());

  return 0;
}
