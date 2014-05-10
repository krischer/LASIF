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


lasifApp.directive('d3Map', function ($window, $log, $aside, $q, $http, $timeout, $location, $alert) {
    return {
        restrict: 'EA',
        scope: false,
        replace: true,
        template: "<div style='width: 100%; height: 100%; margin: 0 auto;'></div>",
        link: function ($scope, element, attrs) {

            // Object collecting all data for the object.
            var data = {
                land_topo: null,
                domain_boundaries_geojson: null,
                all_events: null
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

            $scope.$watch("shownEvents", function (old_value, new_value) {
                if (old_value == new_value) {
                    return;
                }
                redraw();
            });

            // Define canvas including navigation.
            canvas = d3.select(element[0]).append("canvas");
            canvas
                .attr("width", dim.width)
                .attr("height", dim.height)
                .style("width", dim.width / dim.pixelRatio + "px")
                .style("height", dim.height / dim.pixelRatio + "px")

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
                .context(context);

            // Download all necessary data.
            queue()
                .defer(d3.json, "/static/data/world-110m.json")
                .defer(d3.json, "/rest/domain.geojson")
                .defer(d3.json, "/rest/event")
                .await(function (error, world, boundaries, all_events) {
                    if (error) return $log.error(error);

                    // Assign the results of XHR call to the data object.
                    data.land_topo = topojson.feature(world, world.objects.land);
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
                    redraw();
                });

            function redraw() {
                context.clearRect(0, 0, dim.width, dim.height);

                // Draw and fill the land.
                context.lineWidth = .75 * dim.pixelRatio;
                context.strokeStyle = "#000";
                context.fillStyle = "#eee";
                context.beginPath(), path(data.land_topo), context.fill(), context.stroke();

                // Draw the graticule.
                context.lineWidth = .5 * dim.pixelRatio;
                context.strokeStyle = "#ccc";
                context.beginPath(), path(graticule), context.stroke();

                // Draw the domain boundaries.
                context.lineWidth = 2.0 * dim.pixelRatio;
                context.strokeStyle = "#c00";
                context.beginPath(), path(data.domain_boundaries_geojson), context.stroke();

                // Filter out events to only show those that are in $scope.shownEvents.
                var event_list = {type: "GeometryCollection",
                    geometries: _(data.all_events).map(function (i) {
                        if ($scope.shownEvents && $scope.shownEvents != i.event_name) {
                            return null;
                        } else {
                            return i._geometry.circle;
                        }
                    }).compact().value()};
                $log.info(event_list);

                // Draw the events.
                context.lineWidth = 2.0 * dim.pixelRatio;
                context.strokeStyle = "rgba(0, 0, 150, 1.0)";
                context.fillStyle = "rgba(0, 0, 150, 0.3)";
                context.beginPath(), path(event_list), context.fill(), context.stroke();
            };


            canvas.on('click', onClickCanvas)
            function onClickCanvas() {
                // Invert to get longitue/latitude values.
                var point = projection.invert(
                    [d3.event.offsetX * dim.pixelRatio,
                        d3.event.offsetY * dim.pixelRatio]);

                if (_.isNaN(point[0])) {
                    return
                }

                var event = _(data.all_events)
                    .map(function (i) {
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
                    $timeout(function () {
                        $http.get("/rest/event/" + event.event_name, {
                            cache: true
                        }).success(function (data) {
                            asideEl.$scope.stations = data.stations;
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
