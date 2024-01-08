# scripts for plotting multi-channel dice
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from skimage import measure

# Plot a 3D mesh from a binary  3D label alongside the ground truth
def plot3Dmesh(gt, pred, dice, save_path=""):
    gt_verts, gt_faces, gt_normals, gt_values = measure.marching_cubes(gt, 0)
    pred_verts, pred_faces, pred_normals, pred_values = measure.marching_cubes(pred, 0)

    # lighting settings for PlotLy objects
    lighting = dict(ambient=0.5, diffuse=0.5, roughness=0.5, specular=0.6, fresnel=0.8)

    # create the Mesh3d graphical object based on the vertices,
    # faces and values of the original mesh

    gt_x, gt_y, gt_z = gt_verts.T
    gt_I, gt_J, gt_K = gt_faces.T

    pred_x, pred_y, pred_z = pred_verts.T
    pred_I, pred_J, pred_K = pred_faces.T

    gt_mesh = go.Mesh3d(x=gt_x, y=gt_y, z=gt_z,
                        intensity=gt_values,
                        i=gt_I, j=gt_J, k=gt_K,
                        name='Pancreas',
                        lighting=lighting,
                        showscale=False,
                        opacity=1.0,
                        colorscale='magma'
                        )

    pred_mesh = go.Mesh3d(x=pred_x, y=pred_y, z=pred_z,
                        intensity=pred_values,
                        i=pred_I, j=pred_J, k=pred_K,
                        name='Pancreas',
                        lighting=lighting,
                        showscale=False,
                        opacity=1.0,
                        colorscale='magma'
                        )

    # PlotLy figure layout
    layout = go.Layout(
        width=500,
        height=500,
        margin=dict(t=50, l=10, b=10),
    )

    # create figure object
    fig = make_subplots(
        rows=1, cols=2, specs=[[{'type': 'surface'}, {'type': 'surface'}]], subplot_titles=('Ground Truth', 'Prediction'))

    fig.add_trace(gt_mesh, row=1, col=1)
    fig.add_trace(pred_mesh, row=1, col=2)

    fig.update_layout(title_text="Dice score: {0:.2f}".format(dice))
    fig.update_xaxes(visible=False, showticklabels=False)

    # display
    if save_path == "":
        fig.show()
    else:
        fig.write_image(save_path)

