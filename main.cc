#include <easy3d/fileio/resources.h>
#include <easy3d/util/logging.h>
#include <easy3d/renderer/renderer.h>
#include <easy3d/renderer/drawable_points.h>
#include <easy3d/renderer/drawable_lines.h>
#include <easy3d/renderer/drawable_triangles.h>
#include <easy3d/viewer/comp_viewer.h>
#include <easy3d/fileio/surface_mesh_io.h>
#include <easy3d/core/matrix.h>

using namespace easy3d;

struct hyperparameters{
    int MAXSTEP=5000;
    double weight = 0.003;
    double lr = .00001;
} myparams;

struct Edge {
    int s_idx, t_idx;

    Edge(int _s, int _t) : s_idx(_s), t_idx(_t) {};

    Edge() : s_idx(-1), t_idx(-1) {};
};

// Define the edge pair
struct edge_pair {
    Edge e1;
    Edge e2;
};

// Define the line
typedef Vec<3, float> Point3d;

struct Line {
    Point3d p1, p2;
    Point3d dir;

    double length;

    Line(Point3d _p1, Point3d _p2) : p1(_p1), p2(_p2) {
        dir = Point3d();
        dir.x = p2.x - p1.x;
        dir.y = p2.y - p1.y;
        dir.z = p2.z - p1.z;
        length = sqrt(pow(dir.x, 2) + pow(dir.y, 2) + pow(dir.z, 2));
        dir.x = dir.x / length;
        dir.y = dir.y / length;
        dir.z = dir.z / length;
    };

    Line() {
        p1 = Point3d();
        p2 = Point3d();
        dir = Point3d();
        length = 0.0;
    };
};

std::vector<Point3d> Gradient_Optimization(std::vector<Point3d> &vts,
                                           std::vector<edge_pair> orthogonal_pairs,
                                           std::vector<std::vector<int>> &faces,
                                           std::vector<easy3d::Matrix<float>> &Eis) {
    // Obtain the data to vector p
    int n = vts.size();
    easy3d::Matrix<float> X(3 * n, 1, 0.), P(3 * n, 1, 0.), gradX(3 * n, 1, 0.);
    for (int i = 0; i < n; ++i) {
        P(3 * i,0) = vts[i].x;
        P(3 * i + 1,0) = vts[i].y;
        P(3 * i + 2,0) = vts[i].z;
    }

    // Initialize the variable X
    X = P;
    double C = myparams.weight;
    double learning_rate = myparams.lr;
    int step = 0, maxStep = myparams.MAXSTEP;
    while (true) {

        //forward
        easy3d::Matrix<float> EiEjX(3*n,1,0.);
        double regularizer = 0.;

        for (size_t i = 0; i < orthogonal_pairs.size(); i++) {
            easy3d::Matrix<float> EjX(3, 1), EiX(3, 1);
            EjX = Eis[2 * i + 1].transpose() * X;
            EiX = Eis[2 * i].transpose() * X;
            double current_reg = (X.transpose() * Eis[2 * i] * EjX)(0, 0);
            if (current_reg >= 0)
                EiEjX += Eis[2 * i] * EjX + Eis[2 * i + 1] * EiX;
            else
                EiEjX -= Eis[2 * i] * EjX + Eis[2 * i + 1] * EiX;
            regularizer += std::abs(current_reg);
        }

        auto Loss = ((X - P).transpose() * (X - P))(0, 0) + C * regularizer;
        auto objective_Loss = ((X - P).transpose() * (X - P))(0, 0);

        std::cout << "Update step objloss " << step << ": " << objective_Loss
                  << " regularizer "
                  << ": " << regularizer << std::endl;

        if (regularizer > 0)
            gradX = X * 2 - P * 2 + C * EiEjX;
        if (regularizer <= 0)
            gradX = X * 2 - P * 2 - C * EiEjX;

        // update step
        X -= learning_rate * gradX;

        //printout
        if (step == maxStep)
            break;
        step++;
    }
    for (int i = 0; i < vts.size(); i++) {
        vts[i].x = X(3 * i,0);
        vts[i].y = X(3 * i + 1,0);
        vts[i].z = X(3 * i + 2,0);
    }
    return vts;
}

void processInput(std::vector<vec3> &vertices, std::vector<std::vector<int> > &faces, std::vector<Edge> &edges,
                  std::vector<edge_pair> &orthogonal_pairs, std::vector<easy3d::Matrix<float> > &Eis) {

    // Building up adjacency matrix
    bool adjacency[vertices.size()][vertices.size()];
    for (int i = 0; i < vertices.size(); ++i) {
        for (int j = 0; j < vertices.size(); ++j) {
            adjacency[i][j] = false;
        }
    }
    for (auto f: faces) {
        int n_indices = f.size();
        for (int i = 0; i < f.size(); ++i) {
            adjacency[f[i] - 1][f[(i + 1) % n_indices] - 1] = true;
            adjacency[f[(i + 1) % n_indices] - 1][f[i] - 1] = true;
        }
    }

    //Logging the edges from adjacency matrix
    for (int i = 0; i < vertices.size(); ++i) {
        for (int j = i; j < vertices.size(); ++j) {
            if (adjacency[i][j] == true) {
                Edge e(i, j);
                edges.push_back(e);
            }
        }
    }
    // Set the cosine angle threshold
    double thres_cos = 0.1;
    // Cluster the edges into the parallel set and the orthogonal set
    int first_edge_ind = 0, second_edge_ind = 0;
    for (auto it_first = edges.begin(); it_first != std::prev(edges.end()); ++it_first) {
        Edge e1 = *it_first;
        for (auto it_second = std::next(it_first); it_second != edges.end(); ++it_second) {
            Edge e2 = *it_second;

            // construct the lines and the geometry
            Line l1(vertices[e1.s_idx], vertices[e1.t_idx]);
            Line l2(vertices[e2.s_idx], vertices[e2.t_idx]);
            // calculate the dot product, directions have been normalized
            double cos_angle12 = l1.dir.x * l2.dir.x + l1.dir.y * l2.dir.y + l1.dir.z * l2.dir.z;
            // if the cosine of the angle is close to 0, add the pair to orthogonal set
            if (abs(cos_angle12) < thres_cos) {
                edge_pair ep;
                ep.e1 = e1;
                ep.e2 = e2;
                orthogonal_pairs.push_back(ep);

                int n = 3 * vertices.size();

                easy3d::Matrix<float> Ei(3 * vertices.size(), 3,0.);
                easy3d::Matrix<float> Ej(3 * vertices.size(), 3,0.);

                Ei(3 * e1.s_idx, 0) = -1.;
                Ei(3 * e1.s_idx + 1, 1) = -1.;
                Ei(3 * e1.s_idx + 2, 2) = -1.;

                Ei(3 * e1.t_idx, 0) = 1.;
                Ei(3 * e1.t_idx + 1, 1) = 1.;
                Ei(3 * e1.t_idx + 2, 2) = 1.;

                Ej(3 * e2.s_idx, 0) = -1.;
                Ej(3 * e2.s_idx + 1, 1) = -1.;
                Ej(3 * e2.s_idx + 2, 2) = -1.;

                Ej(3 * e2.t_idx, 0) = 1.;
                Ej(3 * e2.t_idx + 1, 1) = 1.;
                Ej(3 * e2.t_idx + 2, 2) = 1.;

                Eis.emplace_back(Ei);
                Eis.emplace_back(Ej);
            }
        }
    }

}

int main() {

    std::vector<std::vector<int> > faces;
    std::vector<easy3d::Matrix<float> > Eis;
    std::vector<Edge> edges;
    std::vector<edge_pair> orthogonal_pairs;
    SurfaceMesh *mesh2 = new SurfaceMesh;
    std::vector<SurfaceMesh::Vertex> new_vts, test;
    std::string model_name = "../data/model1.obj";


    SurfaceMesh *mesh = SurfaceMeshIO::load(model_name);

    auto vertices = mesh->points();
    auto _faces = mesh->faces();


    for (auto f: _faces) {
        std::vector<int> single_face;
        for (auto v: mesh->vertices(f)) {
            single_face.emplace_back(v.idx());
        }
        faces.emplace_back(single_face);
    }

    processInput(vertices, faces, edges, orthogonal_pairs, Eis);
    Gradient_Optimization(vertices, orthogonal_pairs, faces, Eis);
    std::cout << "Points SIZE " << vertices.size() << std::endl;
    //----------- Building the mesh------------------------
    for (auto v: vertices) {
        std::cout << v << std::endl;
        new_vts.emplace_back(mesh2->add_vertex(vec3(v[0], v[1], v[2])));
    }
    for (auto f: faces) {
        test.clear();
        for (int i = 0; i < f.size(); i++) {
            test.emplace_back(new_vts[f[i]]);
        }
        std::cout << "ADD FACE\n";
        mesh2->add_face(test);
    }
    //-------------------------------------------------------
    CompViewer viewer(2, 2, "Shape Regularization Gradient Descent");


    const std::string my_inp = model_name;
    auto input_model = viewer.add_model(my_inp, true);
    if (input_model)
        viewer.view(0, 0).models.push_back(input_model);
    else
        LOG(ERROR) << "failed to load model from file: " << my_inp;

    // ---------------------------------------------------------------------------
    /// setup content for view(0, 1): we show the surface of the sphere model
    auto opt_mesh = viewer.add_model(mesh2, true);
    if (opt_mesh)
        viewer.view(0, 1).models.push_back(opt_mesh);
    else
        LOG(ERROR) << "failed to load model optimized mesh of : " << my_inp;

    // ---------------------------------------------------------------------------
    /// setup content for view(1, 0): we show the wireframe of the sphere model
    auto input_mesh_wireframe = input_model->renderer()->get_lines_drawable("edges");
    input_mesh_wireframe->set_impostor_type(LinesDrawable::CYLINDER);
    input_mesh_wireframe->set_line_width(5);
    input_mesh_wireframe->set_uniform_coloring(vec4(0.7f, 0.7f, 1.0f, 1.0f));
    input_mesh_wireframe->set_visible(true); // by default wireframe is hidden
    viewer.view(1, 0).drawables.push_back(input_mesh_wireframe);

    // ---------------------------------------------------------------------------
    /// setup content for view(1, 1): we show the vertices of the sphere model
    auto opt_mesh_wireframe = opt_mesh->renderer()->get_lines_drawable("edges");
    opt_mesh_wireframe->set_impostor_type(LinesDrawable::CYLINDER);
    opt_mesh_wireframe->set_line_width(5);
    opt_mesh_wireframe->set_uniform_coloring(vec4(0.7f, 0.7f, 1.0f, 1.0f));
    opt_mesh_wireframe->set_visible(true); // by default wireframe is hidden
    viewer.view(1, 1).drawables.push_back(opt_mesh_wireframe);

    // Run the viewer
    return viewer.run();

}
