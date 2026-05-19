import os
import shutil
import numpy as np
import imageio

import trimesh
import polyscope as ps

import logging
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)


def generate_video(image_folder, fps=2, video_path=None):
    # Output video path with .mp4 extension
    if video_path is None:
        video_path = image_folder + '.mp4'

    # Get the list of image file names sorted in order
    images = sorted([img for img in os.listdir(image_folder) 
                     if img.endswith(".png") or img.endswith(".jpg")])

    # Create a video writer using the ffmpeg backend (default for .mp4)
    writer = imageio.get_writer(video_path, fps=fps, codec='libx264')

    # Append each image frame to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = imageio.imread(img_path)
        writer.append_data(frame)
    
    writer.close()

def render_trajectory(
    mesh_dir,
    video_dir,
    mesh_color=(0.11, 0.39, 0.89),
    edge_color=(0.0, 0.0, 0.0),
    edge_width=0.0,
    R=3.5,
    h=2.5,
    N=180,
    show_ground_plane=False,
    ):

    os.makedirs(video_dir, exist_ok=True)

    ps.set_allow_headless_backends(True)
    ps.init()

    ps.set_window_size(1248, 896)

    if show_ground_plane:
        ps.set_ground_plane_mode("tile_reflection")
    else:
        ps.set_ground_plane_mode("shadow_only")
    

    thetas = np.linspace(0, 2*np.pi, N, endpoint=False)

    for fname in sorted(os.listdir(mesh_dir)):
        if not fname.lower().endswith(".obj"):
            continue

        tmp_image_dir = os.path.join(mesh_dir, 'tmp_images')

        out_dir = tmp_image_dir
        os.makedirs(out_dir, exist_ok=True)

        mesh_name = os.path.splitext(fname)[0]
        mesh_path = os.path.join(mesh_dir, fname)

        # Load the mesh
        loaded_mesh = trimesh.load(mesh_path)
        V, F = loaded_mesh.vertices, loaded_mesh.faces

        # Register and render
        ps_mesh = ps.register_surface_mesh(
            mesh_name, V, F,
            color=mesh_color,
            edge_color=edge_color,
            edge_width=edge_width,
            smooth_shade=False,
            material="clay"
        )

        for i, θ in enumerate(thetas):
            cam_pos = (R*np.sin(θ), h, R*np.cos(θ))
            ps.look_at(cam_pos, (0, 0, 0))
            ps.screenshot(os.path.join(out_dir, f"frame_{i:04d}.png"))

        generate_video(out_dir, fps=60, video_path=os.path.join(video_dir, mesh_name + '.mp4'))

        # ps.remove_surface_mesh(mesh_name)
        ps.remove_all_structures()
        shutil.rmtree(tmp_image_dir)

def render_trajectory_to_multiple_dirs(
    mesh_dir,
    video_dir,
    mesh_color=(0.11, 0.39, 0.89),
    edge_color=(0.0, 0.0, 0.0),
    edge_width=0.0,
    R=3.5,
    h=2.5,
    N=180,
    ):

    os.makedirs(video_dir, exist_ok=True)
    
    # first organize the meshes into subdirs
    import shutil

    # the files are named as deformed_mesh_0000_0000.obj, deformed_mesh_0000_0001.obj, ... for first source mesh's deformations
    source_indices = []
    for fname in sorted(os.listdir(mesh_dir)):
        if not fname.lower().endswith(".obj"):
            continue

        source_mesh_idx = fname.split('_')[-2]
        subdir_path = os.path.join(mesh_dir, source_mesh_idx)
        if source_mesh_idx not in source_indices:
            source_indices.append(source_mesh_idx)
            os.makedirs(subdir_path, exist_ok=True)
        
        shutil.move(os.path.join(mesh_dir, fname), os.path.join(subdir_path, fname))

    
    for source_idx in source_indices:
        subdir_path = os.path.join(mesh_dir, source_idx)
        sub_video_dir = os.path.join(video_dir, source_idx + '_videos')
        os.makedirs(sub_video_dir, exist_ok=True)

        render_trajectory(
            subdir_path,
            sub_video_dir,
            mesh_color=mesh_color,
            edge_color=edge_color,
            edge_width=edge_width,
            R=R,
            h=h,
            N=N
        )

def render_all_meshes(
    mesh_dir: str,
    video_path: str,
    screenshot_ext: str = ".png",
    mesh_color=(0.11, 0.39, 0.89),
    edge_color=(0.0, 0.0, 0.0),
    camera_location=(1.0, 1.0, 1.0),
    camera_target=(0.0, 0.0, 0.0),
    edge_width=1.0,
    fps=60,
):
    """
    For each .obj mesh in `mesh_dir`, load it, register it with Polyscope,
    position the camera, take a screenshot, and save it under `image_dir`.
    """

    ps.set_allow_headless_backends(True)
    # ps.set_automatically_compute_scene_extents(False)  # stop auto-updates 
    ps.set_autoscale_structures(False)
    ps.init()

    ps.set_window_size(1248, 896)

    ps.set_ground_plane_mode("shadow_only")
    # ps.set_ground_plane_mode("tile_reflection")

    # Use the desired screenshot file extension
    ps.set_screenshot_extension(screenshot_ext)

    tmp_image_dir = os.path.join(mesh_dir, 'tmp_images')
    os.makedirs(tmp_image_dir, exist_ok=True)

    mesh_list = []
    # Iterate over all OBJ files
    for fname in sorted(os.listdir(mesh_dir)):
        if not fname.lower().endswith(".obj"):
            continue
        mesh_name = os.path.splitext(fname)[0]
        mesh_path = os.path.join(mesh_dir, fname)

        # Load the mesh
        loaded_mesh = trimesh.load(mesh_path)
        V, F = loaded_mesh.vertices, loaded_mesh.faces

        # Register and render
        mesh_list.append(ps.register_surface_mesh(
            mesh_name, V, F,
            color=mesh_color,
            edge_color=edge_color,
            edge_width=edge_width,
            enabled=False
        ))

    ps.look_at(camera_location, camera_target)
    for idx, fname in enumerate(sorted(os.listdir(mesh_dir))):
        if not fname.lower().endswith(".obj"):
            continue

        mesh_name = os.path.splitext(fname)[0]

        mesh_list[idx].set_enabled(True)

        out_path = os.path.join(tmp_image_dir, mesh_name + screenshot_ext)
        ps.screenshot(out_path)

        mesh_list[idx].set_enabled(False)

        ps.look_at(camera_location, camera_target)

    generate_video(tmp_image_dir, fps=fps, video_path=video_path)

    ps.remove_all_structures()
    shutil.rmtree(tmp_image_dir)