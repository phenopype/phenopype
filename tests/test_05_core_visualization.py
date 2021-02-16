#%% modules

import phenopype as pp

#%% test

def test_select_canvas(project_container):
    pp.visualization.select_canvas(project_container, canvas="red", multi=True)
    assert not (project_container.image_copy==project_container.canvas).all()

def test_draw_contours(project_container):
    pp.visualization.draw_contours(project_container, label=True, mark_holes=True,
                                   fill=0.5, skeleton=True)
    assert not (project_container.image_copy==project_container.canvas).all()

def test_draw_landmarks(project_container):
    pp.visualization.draw_landmarks(project_container)
    assert not (project_container.image_copy==project_container.canvas).all()

def test_draw_masks(project_container):
    pp.visualization.draw_masks(project_container, label=True)
    assert not (project_container.image_copy==project_container.canvas).all()

def test_draw_polylines(project_container):
    pp.visualization.draw_polylines(project_container)
    assert not (project_container.image_copy==project_container.canvas).all()
