
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

using namespace dealii;

template <int dim>
void
run(const unsigned int s, const unsigned int fe_degree, const unsigned int n_components = 1)
{
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

  matrix_free->reinit(mapping, dof_handler, constraints, QGaussLobatto<1>(fe_degree + 1), mf_data);

  {
    const auto &              di = matrix_free->get_dof_info(0);
    const auto &              ti = matrix_free->get_task_info();
    std::vector<unsigned int> distances(di.vector_partitioner->locally_owned_size() / 64 + 1,
                                        numbers::invalid_unsigned_int);
    for (unsigned int id =
           di.cell_loop_pre_list_index[ti.partition_row_index[ti.partition_row_index.size() - 2]];
         id <
         di.cell_loop_pre_list_index[ti.partition_row_index[ti.partition_row_index.size() - 2] + 1];
         ++id)
      for (unsigned int a = di.cell_loop_pre_list[id].first; a < di.cell_loop_pre_list[id].second;
           a += 64)
        distances[a / 64] = 0;
    for (unsigned int part = 0; part < ti.partition_row_index.size() - 2; ++part)
      {
        for (unsigned int i = ti.partition_row_index[part]; i < ti.partition_row_index[part + 1];
             ++i)
          {
            // std::cout << "pre ";
            for (unsigned int id = di.cell_loop_pre_list_index[i];
                 id != di.cell_loop_pre_list_index[i + 1];
                 ++id)
              {
                for (unsigned int a = di.cell_loop_pre_list[id].first;
                     a < di.cell_loop_pre_list[id].second;
                     a += 64)
                  distances[a / 64] = i;
                // std::cout << id << "[" << di.cell_loop_pre_list[id].first << ","
                //          << di.cell_loop_pre_list[id].second << ") ";
              }
            // std::cout << std::endl;
            // std::cout << "post ";
            for (unsigned int id = di.cell_loop_post_list_index[i];
                 id != di.cell_loop_post_list_index[i + 1];
                 ++id)
              {
                for (unsigned int a = di.cell_loop_post_list[id].first;
                     a < di.cell_loop_post_list[id].second;
                     a += 64)
                  distances[a / 64] = i - distances[a / 64];
                // std::cout << id << "[" << di.cell_loop_post_list[id].first << ","
                //          << di.cell_loop_post_list[id].second << ") ";
              }
            // std::cout << std::endl;
          }
        // std::cout << std::endl;
      }
    for (unsigned int id =
           di.cell_loop_post_list_index[ti.partition_row_index[ti.partition_row_index.size() - 2]];
         id <
         di.cell_loop_post_list_index[ti.partition_row_index[ti.partition_row_index.size() - 2] +
                                      1];
         ++id)
      for (unsigned int a = di.cell_loop_post_list[id].first; a < di.cell_loop_post_list[id].second;
           a += 64)
        distances[a / 64] =
          ti.partition_row_index[ti.partition_row_index.size() - 2] - distances[a / 64];
    std::map<unsigned int, unsigned int> count;
    for (const auto a : distances)
      if (a != numbers::invalid_unsigned_int)
        count[a]++;

    unsigned int max_liveliness = 0;

    for (const auto a : count)
      max_liveliness = std::max(a.first, max_liveliness);

    max_liveliness = Utilities::MPI::max(max_liveliness, MPI_COMM_WORLD);

    std::vector<double> temp_(max_liveliness, 0);
    std::vector<double> temp(max_liveliness, 0);

    for (const auto a : count)
      temp_[a.first] = a.second;

    Utilities::MPI::sum(temp_, MPI_COMM_WORLD, temp);

    for (unsigned int i = 1; i < temp.size(); ++i)
      temp[i] += temp[i - 1];

    for (unsigned int i = 0; i < temp.size(); ++i)
      temp[i] /= temp.back();

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::cout << "s=" << s << std::endl;
        std::cout << "n_dofs=" << dof_handler.n_dofs() << std::endl;

        for (unsigned int i = 0; i < temp.size(); ++i)
          std::cout << i << " " << temp[i] * 100 << std::endl;
        std::cout << std::endl;
      }
  }
}



int
main(int argc, char **argv)
{
  // mpirun -np 40 ./benchmark_range/bench 5 39
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  AssertThrow(argc > 2, ExcNotImplemented());

  const unsigned int degree  = std::atoi(argv[1]);
  const unsigned int n_steps = std::atoi(argv[2]);

  run<3>(n_steps, degree);
  return 0;
}
