#include "meshview/meshview.hpp"
#include "meshview/meshview_imgui.hpp"

#include "Avatar.h"
#include "Util.h"

int main(int argc, char** argv) {
    ark::AvatarModel model;
    ark::Avatar ava(model);

    ava.update();

    meshview::Viewer viewer;
    meshview::Triangles faces = model.mesh.transpose().cast<meshview::Index>();

    // Main body model
    auto& smpl_mesh = viewer.add_mesh(ava.cloud.transpose().cast<float>(),
                                      faces, 0.8f, 0.5f, 0.6f);

    // LBS weights color visualization
    auto& smpl_mesh_lbs =
        viewer
            .add_mesh(ava.cloud.transpose().cast<float>(), faces,
                      /* use vertex-based colorization, by passing n-by-3
                         matrix as 3rd argument */
                      (ark::util::paletteColorTable(model.numJoints()) *
                       model.weights.cast<float>())
                          .transpose())
            .translate(Eigen::Vector3f(2.0f, 0.f, 0.f));

    // Joints
    std::vector<meshview::Mesh*> joint_spheres;
    // Lines are 'point clouds' with lines=true, sorry for weirdness
    std::vector<meshview::PointCloud*> joint_lines;
    const Eigen::Vector3f joint_offset(-2.f, 0.f, 0.f);
    for (size_t i = 0; i < model.numJoints(); ++i) {
        auto joint_pos = ava.jointPos.col(i).cast<float>();
        joint_spheres.push_back(
            &viewer
                 .add_sphere(Eigen::Vector3f::Zero(), 0.01f,
                             Eigen::Vector3f(1.f, 0.5f, 0.0f))
                 .translate(joint_pos + joint_offset));
        if (i) {
            auto parent_pos = ava.jointPos.col(model.parent(i)).cast<float>();
            joint_lines.push_back(
                &viewer
                     .add_line(joint_pos, parent_pos,
                               Eigen::Vector3f(0.4f, 0.5f, 0.8f))
                     .translate(joint_offset));
        }
    }

    bool updated = false;
    auto update = [&]() {
        ava.update();
        // Update the mesh on-the-fly (send to GPU)
        smpl_mesh.verts_pos().noalias() = ava.cloud.transpose().cast<float>();
        smpl_mesh_lbs.verts_pos().noalias() =
            ava.cloud.transpose().cast<float>();
        for (size_t i = 0; i < model.numJoints(); ++i) {
            auto joint_pos = ava.jointPos.col(i).cast<float>();
            joint_spheres[i]->set_translation(joint_pos + joint_offset);
            if (i) {
                auto parent_pos =
                    ava.jointPos.col(model.parent(i)).cast<float>();
                joint_lines[i - 1]->verts_pos().row(0).noalias() =
                    joint_pos.transpose();
                joint_lines[i - 1]->verts_pos().row(1).noalias() =
                    parent_pos.transpose();
            }
        }
        updated = true;
    };
    Eigen::Vector3f p;
    p.setZero();
    Eigen::VectorXf r(3 * ava.r.size());
    r.setZero();
    Eigen::VectorXf w(ava.w.rows());
    w.setZero();

    viewer.on_open = []() { ImGui::GetIO().IniFilename = nullptr; };
    viewer.on_gui = [&]() {
        updated = false;
        // * GUI code
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(500, 360), ImGuiCond_Once);
        ImGui::Begin("Model Parameters", NULL);
        ImGui::Text("Model: %s", model.MODEL_DIR.c_str());
        ImGui::TextUnformatted("Press h for help");
        ImGui::TextUnformatted("Reset: ");
        ImGui::SameLine();
        if (ImGui::Button("Trans##ResetTrans")) {
            ava.p.setZero();
            update();
        }
        ImGui::SameLine();
        if (ImGui::Button("Pose##ResetPose")) {
            for (auto& q : ava.r) q.setIdentity();
            update();
        }
        ImGui::SameLine();
        if (ImGui::Button("Shape##ResetShape")) {
            ava.w.setZero();
            update();
        }

        if (ImGui::SliderFloat3("translation", p.data(), -5.f, 5.f)) {
            ava.p = p.cast<double>();
            update();
        }
        if (ImGui::TreeNode("Pose")) {
            const int STEP = 10;
            for (int j = 0; j < model.numJoints(); j += STEP) {
                int end_idx = std::min(j + STEP, model.numJoints());
                if (ImGui::TreeNode(("Angle axis " + std::to_string(j) + " - " +
                                     std::to_string(end_idx - 1))
                                        .c_str())) {
                    for (int i = j; i < end_idx; ++i) {
                        if (ImGui::SliderFloat3(
                                (std::string(std::to_string(i) + "##joint") +
                                 std::to_string(i))
                                    .c_str(),
                                r.data() + i * 3, -1.6f, 1.6f)) {
                            ava.r[i] = ark::util::rodrigues<float>(
                                           Eigen::Map<Eigen::Vector3f>(
                                               r.data() + i * 3))
                                           .cast<double>();
                            update();
                        }
                    }
                    ImGui::TreePop();
                }
            }
            ImGui::TreePop();
        }
        if (ImGui::TreeNode("Shape")) {
            for (size_t i = 0; i < model.numShapeKeys(); ++i) {
                if (ImGui::SliderFloat(
                        (std::string("shape") + std::to_string(i)).c_str(),
                        w.data() + i, -5.f, 5.f)) {
                    ava.w[i] = (double)w[i];
                    update();
                }
            }
            ImGui::TreePop();
        }
        ImGui::End();  // Model Parameters

        ImGui::SetNextWindowPos(ImVec2(10, 395), ImGuiCond_Once);
        ImGui::SetNextWindowSize(ImVec2(500, 100), ImGuiCond_Once);
        ImGui::Begin("Camera and Rendering", NULL);

        if (ImGui::Button("Reset view")) {
            viewer.camera.reset_view();
        }
        ImGui::SameLine();
        if (ImGui::Button("Reset projection")) {
            viewer.camera.reset_proj();
        }
        ImGui::SameLine();
        ImGui::Checkbox("wireframe", &viewer.wireframe);

        if (ImGui::TreeNode("View")) {
            if (ImGui::SliderFloat3("cen_of_rot",
                                    viewer.camera.center_of_rot.data(), -5.f,
                                    5.f))
                viewer.camera.update_view();
            if (ImGui::SliderFloat("radius", &viewer.camera.dist_to_center,
                                   0.01f, 10.f))
                viewer.camera.update_view();
            if (ImGui::DragFloat("yaw", &viewer.camera.yaw))
                viewer.camera.update_view();
            if (ImGui::DragFloat("pitch", &viewer.camera.pitch))
                viewer.camera.update_view();
            if (ImGui::DragFloat("roll", &viewer.camera.roll))
                viewer.camera.update_view();
            if (ImGui::SliderFloat3("world_up", viewer.camera.world_up.data(),
                                    -5.f, 5.f))
                viewer.camera.update_view();
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Projection")) {
            if (ImGui::SliderFloat("fovy", &viewer.camera.fovy, 0.01f, 1.5f))
                viewer.camera.update_proj();
            if (ImGui::SliderFloat("z_close", &viewer.camera.z_close, 0.01f,
                                   10.f))
                viewer.camera.update_proj();
            if (ImGui::SliderFloat("z_far", &viewer.camera.z_far, 11.f, 5000.f))
                viewer.camera.update_proj();
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Lighting")) {
            if (ImGui::SliderFloat3("pos", viewer.light_pos.data(), -4.f, 4.f))
                if (ImGui::SliderFloat3(
                        "ambient", viewer.light_color_ambient.data(), 0.f, 1.f))
                    if (ImGui::SliderFloat3("diffuse",
                                            viewer.light_color_diffuse.data(),
                                            0.f, 1.f))
                        if (ImGui::SliderFloat3(
                                "specular", viewer.light_color_specular.data(),
                                0.f, 1.f))
                            ImGui::TreePop();
        }

        ImGui::End();  // Camera and Rendering
        // Return true if updated to indicate mesh data has been changed
        // the viewer will update the GPU buffers automatically
        return updated;
    };

    viewer.show();
}
