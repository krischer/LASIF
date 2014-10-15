from lasif import domain
domain.RectangularSphericalSection(
    min_latitude=-10,
    max_latitude=10,
    min_longitude=-10,
    max_longitude=10,
    min_depth_in_km=0,
    max_depth_in_km=1440,
    boundary_width_in_degree=2.5,
    rotation_axis=[1.0, 1.0, 0.2],
    rotation_angle_in_degree=-65.0).plot(plot_simulation_domain=True)