var lasifApp = angular.module("LASIFApp");



// Helper function calculating the dimension of everything.
var getDimensions = function (element) {
    var width = parseInt(d3.select(element[0]).style('width'));
    var height = parseInt(d3.select(element[0]).style('height'));
    // Important for high DPI displays.
    var pixelRatio = window.devicePixelRatio || 1;
    var dx = 2.5 * pixelRatio;

    // Scale with the pixel ratio as we are using canvas.
    width *= pixelRatio;
    height *= pixelRatio;

    return {width: width, height: height, pixelRatio: pixelRatio, dx: dx};
};


lasifApp.directive('d3Map', function ($window, $log, $aside, $q, $http, $timeout, $location, $alert, $modal) {
    return {
        restrict: 'EA',
        scope: false,
        replace: true,
        template: "<div style='width: 100%; height: 100%; margin: 0 auto;'></div>",
        link: function ($scope, element, attrs) {

            // Object collecting all data for the object.
            var data = {
                land_topo: undefined,
                countries_topo: undefined,
                domain_boundaries_geojson: undefined,
                all_events: undefined,
                stations: {}
            };

            // Function wide variables.
            var dim;
            var projection;
            var graticule;
            var canvas;
            var context;
            var land;
            var domain_boundaries;
            var event_list;
            var path;

            dim = getDimensions(element);

            projection = d3.geo.orthographic()
                .clipAngle(90)
                .translate([dim.width / 2, dim.height / 2])
                .scale(400 * dim.pixelRatio)
                .precision(1)
                .clipExtent([
                    [dim.dx, dim.dx],
                    [dim.width - dim.dx, dim.height - dim.dx]
                ])
                .rotate([40.7, -26.8])

            graticule = d3.geo.graticule().step([5, 5]).extent([
                [-180, -85],
                [180, 85 + 1e-6]
            ])();

            function update_stations() {
                if (!$scope.shownEvents) {
                    redraw();
                    return
                }
                $http.get("/rest/event/" + $scope.shownEvents, {
                    cache: true
                }).success(function (d) {
                    data.stations["stations"] = d.stations;
                    // Build geojson to ease plotting.
                    data.stations["geojson"] = {
                        type: "MultiPoint",
                        coordinates: _.map(d.stations, function (i) {
                            return [i.longitude, i.latitude]
                        })
                    };
                    // A further list containing raypath endpoints.
                    data.stations["raypaths"] = {
                        type: "MultiLineString",
                        coordinates: _.map(d.stations, function (i) {
                            return [
                                [d.longitude, d.latitude],
                                [i.longitude, i.latitude]
                            ]
                        })
                    };
                    // Build a kdTree to enable fast nearest neighbour
                    // searches over the stations.
                    data.stations["kdTree"] = new kdTree(
                        d.stations,
                        function (a, b) {
                            return d3.geo.distance(
                                [a.longitude, a.latitude],
                                [b.longitude, b.latitude])
                        }, ["longitude", "latitude"]);
                    redraw();
                })
            }

            $scope.$watch("shownEvents", function (new_value, old_value) {

                if (old_value == new_value) {
                    return;
                }

                if (new_value) {
                    update_stations()
                }
                else {
                    data.stations = {};
                    redraw();
                }

            });

            // Define canvas including navigation.
            canvas = d3.select(element[0]).append("canvas");
            canvas
                .attr("width", dim.width)
                .attr("height", dim.height)
                .style("width", dim.width / dim.pixelRatio + "px")
                .style("height", dim.height / dim.pixelRatio + "px")
                .style("position", "relative")

                .call(d3.behavior.zoom()
                    .scale(projection.scale())
                    .scaleExtent([100 * dim.pixelRatio, Infinity])
                    .on("zoom", function () {
                        projection.scale(d3.event.scale);
                        redraw();
                    }))
                .call(d3.behavior.drag()
                    .origin(function () {
                        var r = projection.rotate(), s = .004 * projection.scale();
                        return {x: s * r[0], y: -s * r[1]};
                    })
                    .on("drag", function () {
                        dragging = true;
                        var s = .004 * projection.scale();
                        projection.rotate(initial = [d3.event.x / s, -d3.event.y / s]);
                        redraw();
                    })
                    .on("dragend", function () {
                        dragging = false;
                    }));
            context = canvas.node().getContext("2d");

            path = d3.geo.path()
                .projection(projection)
                .context(context)
                .pointRadius(2.25 * dim.pixelRatio);

            // Download all necessary data.
            queue()
                .defer(d3.json, "/static/data/world-110m.json")
                .defer(d3.json, "/rest/domain.geojson")
                .defer(d3.json, "/rest/event")
                .await(function (error, world, boundaries, all_events) {
                    if (error) return $log.error(error);

                    // Assign the results of XHR call to the data object.
                    data.land_topo = topojson.feature(world, world.objects.land);
                    data.countries_topo = topojson.feature(world, world.objects.countries);
                    data.domain_boundaries_geojson = boundaries;
                    data.all_events = all_events.events;

                    // Calculate a geometrical representation of all events
                    // that can be plotted with D3.
                    _.forEach(data.all_events, function (event) {
                        var radius = Math.pow(event["magnitude"], 2.5) / 50;
                        var point = [event["longitude"], event["latitude"]];
                        var circle = d3.geo.circle()
                            .angle(radius)
                            .origin(function (x, y) {
                                return [x, y]
                            })
                            (point[0], point[1]);
                        event._geometry = {
                            circle: circle,
                            radius: radius,
                        };
                    });
                    update_stations();

                    // Center the map on the boundary region. The rotation is the negative location.
                    var centroid = d3.geo.centroid(boundaries);
                    projection.rotate([-1.0 * centroid[0], -1.0 * centroid[1]]);
                    redraw();
                });

            function redraw() {
                // Uses the following colorscheme:
                // https://kuler.adobe.com/Orange-on-olive-color-theme-2227/
                //
                // #C03000: Red             | Domain Boundaries
                // #B4AF91: Lightest Green  | Raypaths
                // #787746: ...             | Stations
                // #40411E: ...             | Events
                // #32331D: Darkest Green   | Currently unused
                //
                // Rest of the map consists of black, white and gray colors.

                context.clearRect(0, 0, dim.width, dim.height);

                // Draw and fill the land.
                context.lineWidth = .75 * dim.pixelRatio;
                context.fillStyle = "#ddd";
                context.beginPath(), path(data.land_topo), context.fill();

                // Draw country border on top of it.
                context.lineWidth = 1.5 * dim.pixelRatio;
                context.strokeStyle = "#fff";
                context.beginPath(), path(data.countries_topo), context.stroke();

                // Draw the graticule.
                context.lineWidth = .5 * dim.pixelRatio;
                context.strokeStyle = "#ccc";
                context.beginPath(), path(graticule), context.stroke();

                // Draw the domain boundaries.
                context.lineWidth = 2.0 * dim.pixelRatio;
                context.strokeStyle = "#C03000";
                context.beginPath(), path(data.domain_boundaries_geojson),
                    context.stroke();

                // Draw the raypaths if they are available.
                if (data.stations && data.stations.raypaths) {
                    context.lineWidth = 0.5 * dim.pixelRatio;
                    context.strokeStyle = "rgba(180, 175, 145, 50)";
                    context.beginPath(), path(data.stations.raypaths),
                        context.stroke();
                }

                // Draw the stations if they are available.
                if (data.stations && data.stations.geojson) {
//                    context.fillStyle = "rgba(133, 199, 58, 100)";
                    context.fillStyle = "#787746";
                    context.beginPath(), path(data.stations.geojson),
                        context.fill();
                }

                // Filter out events to only show those that are in $scope.shownEvents.
                var event_list = {type: "GeometryCollection",
                    geometries: _(data.all_events).map(function (i) {
                        if ($scope.shownEvents && $scope.shownEvents != i.event_name) {
                            return null;
                        } else {
                            return i._geometry.circle;
                        }
                    }).compact().value()};

                // Actually draw the events.
                context.lineWidth = 2.0 * dim.pixelRatio;
                context.strokeStyle = "rgba(64, 65, 30, 1.0)";
                context.fillStyle = "rgba(64, 65, 30, 0.5)";
                context.beginPath(), path(event_list), context.fill(),
                    context.stroke();
            };

            var scope;

            canvas.on('click', onClickCanvas)
            function onClickCanvas() {
                // Compatibility with Firefox and Chrome.
                // See http://stackoverflow.com/a/11334572
                var e = d3.event;
                var x = e.offsetX == undefined ? e.layerX : e.offsetX;
                var y = e.offsetY == undefined ? e.layerY : e.offsetY;

                // Invert to get longitue/latitude values.
                var point = projection.invert(
                    [x * dim.pixelRatio, y * dim.pixelRatio]);

                if (data.stations && data.stations["kdTree"]) {
                    var nearest_point = data.stations["kdTree"].nearest(
                        {longitude: point[0], latitude: point[1]}
                        , 1)[0];

                    // Check if the nearest point is within a 5 pixel radius.
                    var projected_point = projection(
                        [nearest_point[0].longitude,
                            nearest_point[0].latitude]);

                    var distance = Math.sqrt(
                        Math.pow((x * dim.pixelRatio) - projected_point[0], 2),
                        Math.pow((y * dim.pixelRatio) - projected_point[1], 2));
                    if (distance <= 5 * dim.pixelRatio) {
                        var modal = $modal({
                            title: nearest_point[0].station_name,
                            template: "/static/templates/station_waveform_plot.tpl.html",
                            persist: false,
                            show: true});
                        // Set some information about the station and event.
                        modal.$scope.event_name = $scope.shownEvents;
                        modal.$scope.station = nearest_point[0];
                    }
                }

                if (_.isNaN(point[0])) {
                    return
                }

                var event = _(data.all_events)
                    .map(function (i) {
                        if ($scope.shownEvents && $scope.shownEvents != i.event_name) {
                            return
                        }

                        var dist = d3.geo.distance(
                            point,
                            [i.longitude, i.latitude]) / Math.PI * 180;
                        i.distance = dist;
                        return dist <= i._geometry.radius
                            ? i : null;
                    })
                    .compact()
                    .min('distance').value();

                if (_.isUndefined(event) || event === Infinity) {
                    return
                }

                // Show an aside element with some event details.
                var aside = $aside({
                    title: "Event Details",
                    template: "/static/templates/event_detail.tpl.html",
                    persist: false,
                    show: true
                });

                $q.when(aside).then(function (asideEl) {
                    // Function to change the event view.
                    var plot_event = function () {
                        $location.path("/map/" + event.event_name);
                    };

                    asideEl.$scope.event = event;
                    asideEl.$scope.plot_event = plot_event;
                    asideEl.$scope.stations_downloaded = false;
                    $timeout(function () {
                        $http.get("/rest/event/" + event.event_name, {
                            cache: true
                        }).success(function (data) {
                            asideEl.$scope.stations = data.stations;
                            asideEl.$scope.stations_downloaded = true;
                        })
                    }, 200);
                })
            };

            // Delay resize a bit as it is fairly expensive.
            d3.select(window).on('resize', resizeDelay);
            var delayIt;

            function resizeDelay() {
                clearTimeout(delayIt);
                delayIt = setTimeout(resize, 100);
            }

            function resize() {
                // Update dimensions.
                dim = getDimensions(element)

                // update projection
                projection
                    .translate([dim.width / 2, dim.height / 2])
                    .clipExtent([
                        [dim.dx, dim.dx],
                        [dim.width - dim.dx, dim.height - dim.dx]
                    ]);

                // resize the map container
                canvas
                    .attr('width', dim.width + 'px')
                    .attr('height', dim.height + 'px');
                canvas
                    .style('width', dim.width / dim.pixelRatio + 'px')
                    .style('height', dim.height / dim.pixelRatio + 'px');

                redraw();
            }
        }
    };
});
