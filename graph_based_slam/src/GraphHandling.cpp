// project headers
#include "graph_based_slam/GraphHandling.h"
#include "graph_based_slam/ConeData.h"

// C++ standard headers
#include <string>
#include <vector>

// ROS headers
#include <rclcpp/rclcpp.hpp>

// g2o headers
#include <g2o/core/factory.h>
#include <g2o/core/linear_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
// #include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/block_solver.h>
#include "graph_based_slam/icr_edge.hpp"

G2O_USE_TYPE_GROUP(slam2d)
// G2O_USE_OPTIMIZATION_LIBRARY(eigen)

// using namespace std;

// GraphHandling::GraphHandling(const rclcpp::NodeOptions &options) : Node("graph_handling_node", options)
GraphHandling::GraphHandling(const std::string &solver_type)
{
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<-1, -1>> SlamBlockSolver;
    typedef g2o::LinearSolverCSparse<SlamBlockSolver::PoseMatrixType> SlamLinearSolver;

    slam_graph.reset(new g2o::SparseOptimizer());
    g2o::SparseOptimizer *slam_graph = dynamic_cast<g2o::SparseOptimizer *>(this->slam_graph.get());

    // optimize with Levenberg-Marquardt
    // std::unique_ptr<g2o::BlockSolverX::LinearSolverType> linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    // g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver)));
    auto linearSolver = std::make_unique<SlamLinearSolver>();
    linearSolver->setBlockOrdering(false);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<SlamBlockSolver>(std::move(linearSolver)));
    // g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(g2o::make_unique<SlamBlockSolver>(std::move(linearSolver)));

    slam_graph->setAlgorithm(solver);

    if (!slam_graph->solver())
    {
        std::cerr << std::endl;
        std::cerr << "Error : failed to allocate solver!!" << std::endl;
        std::cerr << "-------------" << std::endl;
        std::cin.ignore(1);
        return;
    }
}

GraphHandling::~GraphHandling()
{
    slam_graph.reset();
}

int GraphHandling::num_vertices() const
{
    return slam_graph->vertices().size();
}

int GraphHandling::num_edges() const
{
    return slam_graph->edges().size();
}

g2o::VertexSE2 *GraphHandling::createPoseVertex(const Eigen::Isometry2d &pose)
{
    g2o::VertexSE2 *pose_node(new g2o::VertexSE2());
    pose_node->setId(static_cast<int>(slam_graph->vertices().size()));
    pose_node->setEstimate(pose);

    return pose_node;
}

g2o::VertexSE2 *GraphHandling::addPoseVertex(g2o::VertexSE2 *pose_vertex)
{
    slam_graph->addVertex(pose_vertex);
    last_added_pose = pose_vertex;
    return pose_vertex;
}

g2o::VertexPointXY *GraphHandling::createLandmarkVertex(const Eigen::Vector2d &xy_coords)
{
    g2o::VertexPointXY *landmark_node(new g2o::VertexPointXY());
    landmark_node->setId(static_cast<int>(slam_graph->vertices().size()));
    landmark_node->setEstimate(xy_coords);

    return landmark_node;
}

g2o::VertexPointXY *GraphHandling::addLandmarkVertex(g2o::VertexPointXY *land_vertex)
{
    slam_graph->addVertex(land_vertex);

    return land_vertex;
}

g2o::EdgeSE2 *GraphHandling::createPoseEdge(g2o::VertexSE2 *node1, g2o::VertexSE2 *node2, const Eigen::Isometry2d &rel_pose, const Eigen::MatrixXd &info_matrix)
{
    g2o::EdgeSE2 *pose_edge(new g2o::EdgeSE2());
    pose_edge->setMeasurement(rel_pose);
    pose_edge->setInformation(info_matrix);
    pose_edge->vertices()[0] = node1;
    pose_edge->vertices()[1] = node2;

    return pose_edge;
}

g2o::EdgeSE2 *GraphHandling::addPoseEdge(g2o::EdgeSE2 *pose_edge)
{
    /**/
    g2o::Edge2ICR *icr_edge(new g2o::Edge2ICR());
    icr_edge->setMeasurement(pose_edge->measurement());

    icr_edge->setInformation(Eigen::Matrix2d::Identity() * 0.5);
    icr_edge->vertices()[0] = pose_edge->vertices()[0];
    icr_edge->vertices()[1] = pose_edge->vertices()[1];
    /**/

    slam_graph->addEdge(pose_edge);
    slam_graph->addEdge(icr_edge);

    pose_edge->computeError();
    // pose_edge->linearizeOplus();

    return pose_edge;
}

g2o::EdgeSE2PointXY *GraphHandling::createLandmarkEdge(g2o::VertexSE2 *pose_vertex, g2o::VertexPointXY *land_vertex, const Eigen::Vector2d &xy_coords, const Eigen::MatrixXd &info_matrix)
{
    g2o::EdgeSE2PointXY *landmark_edge(new g2o::EdgeSE2PointXY());
    landmark_edge->setMeasurement(xy_coords);
    landmark_edge->setInformation(info_matrix);
    landmark_edge->vertices()[0] = pose_vertex;
    landmark_edge->vertices()[1] = land_vertex;

    return landmark_edge;
}

g2o::EdgeSE2PointXY *GraphHandling::addLandmarkEdge(g2o::EdgeSE2PointXY *land_edge)
{

    slam_graph->addEdge(land_edge);

    land_edge->computeError();
    // std::cout << "Val = " << land_edge->chi2() << std::endl;
    // land_edge->linearizeOplus();

    return land_edge;
}

int GraphHandling::globalOptimization(int num_iterations)
{
    g2o::SparseOptimizer *slam_graph = dynamic_cast<g2o::SparseOptimizer *>(this->slam_graph.get());
    if (slam_graph->edges().size() < 10)
    {
        return -1;
    }

    std::cout << std::endl;
    std::cout << "--- Global pose graph optimization ---" << std::endl;
    std::cout << "nodes: " << slam_graph->vertices().size() << "   edges: " << slam_graph->edges().size() << std::endl;
    std::cout << "optimizing... " << std::flush;

    slam_graph->initializeOptimization();
    slam_graph->setVerbose(true);

    double chi2 = slam_graph->chi2();

    int iterations = slam_graph->optimize(num_iterations);

    std::cout << "Optimization completed!" << std::endl;
    std::cout << "iterations: " << iterations << " / " << num_iterations << std::endl;
    std::cout << "chi2: (before)" << chi2 << " -> (after)" << slam_graph->chi2() << std::endl;

    return iterations;
}

int GraphHandling::localOptimization(int num_iterations, g2o::HyperGraph::EdgeSet edges_to_optim)
{
    g2o::SparseOptimizer *slam_graph = dynamic_cast<g2o::SparseOptimizer *>(this->slam_graph.get());
    if (slam_graph->edges().size() < 5)
    {
        return -1;
    }

    std::cout << std::endl;
    std::cout << "--- Local pose graph optimization ---" << std::endl;
    std::cout << "nodes: " << slam_graph->vertices().size() << "   edges: " << slam_graph->edges().size() << std::endl;
    std::cout << "optimizing... " << std::flush;

    slam_graph->initializeOptimization(edges_to_optim);
    slam_graph->setVerbose(true);

    double chi2 = slam_graph->chi2();

    int iterations = slam_graph->optimize(num_iterations);

    std::cout << "Optimization completed!" << std::endl;
    std::cout << "iterations: " << iterations << " / " << num_iterations << std::endl;
    std::cout << "chi2: (before)" << chi2 << " -> (after)" << slam_graph->chi2() << std::endl;

    return iterations;
}

void GraphHandling::saveGraph(const std::string &filename)
{
    // TODO: check save on file
    g2o::SparseOptimizer *slam_graph = dynamic_cast<g2o::SparseOptimizer *>(this->slam_graph.get());
    std::ofstream output_fs(filename);
    slam_graph->save(output_fs);
}

bool GraphHandling::loadGraph(const std::string &filename)
{
    // TODO: check load from file
    g2o::SparseOptimizer *slam_graph = dynamic_cast<g2o::SparseOptimizer *>(this->slam_graph.get());

    std::ifstream input_fs(filename);
    if (slam_graph->load(input_fs))
    {
        return true;
    }
    return false;
}

g2o::SparseOptimizer *GraphHandling::getGraph()
{
    g2o::SparseOptimizer *slam_graph = dynamic_cast<g2o::SparseOptimizer *>(this->slam_graph.get());
    return slam_graph;
}

g2o::HyperGraph::VertexIDMap GraphHandling::getNodes()
{
    return slam_graph->vertices();
}

g2o::HyperGraph::EdgeSet GraphHandling::getEdges()
{
    return slam_graph->edges();
}

g2o::VertexSE2 *GraphHandling::getLastPoseNode()
{
    return this->last_added_pose;
}

g2o::VertexPointXY *GraphHandling::getLastLandmarkNode()
{
    return this->last_added_cone;
}
