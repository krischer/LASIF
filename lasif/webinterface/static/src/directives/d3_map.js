var lasifApp = angular.module("LASIFApp");

lasifApp.directive('d3Map', function ($window, $log) {
    return {
        restrict: 'EA',
        scope: {},
        replace: true,
        template: "<div style='width: 100%; height: 100%; margin: 0 auto;'></div>",
        link: function ($scope, element, attrs) {

            var mapRatio = 0.1;
            var width = parseInt(d3.select(element[0]).style('width'));
            var height = parseInt(d3.select(element[0]).style('height'));

            // Important for high DPI displays.
            var pixelRatio = window.devicePixelRatio || 1;

            // Scale as we are using canvas for displaying.
            width *= pixelRatio;
            height *= pixelRatio;

            var dx = 2.5 * pixelRatio;

            var projection = d3.geo.orthographic()
                .clipAngle(90)
                .translate([width / 2, height / 2])
                .scale(400 * pixelRatio)
                .precision(1)
                .clipExtent([
                    [dx, dx],
                    [width - dx, height - dx]
                ])
                .rotate([40.7, -26.8])

            var graticule = d3.geo.graticule().step([5, 5]).extent([
                [-180, -85],
                [180, 85 + 1e-6]
            ])();

            var canvas = d3.select(element[0]).append("canvas")
                .attr("width", width)
                .attr("height", height)
                .style("width", width / pixelRatio + "px")
                .style("height", height / pixelRatio + "px")
                .style("box-shadow", "1px 1px 7px 0px rgba(50, 50, 50, 0.32)")

                .call(d3.behavior.zoom()
                    .scale(projection.scale())
                    .scaleExtent([100 * pixelRatio, Infinity])
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
                        t0 = Date.now();
                    }));

            var context = canvas.node().getContext("2d");
            var land;
            var domain_boundaries;
            var event_list;

            var path = d3.geo.path()
                .projection(projection)
                .context(context);

            queue()
                .defer(d3.json, "/static/data/world-110m.json")
                .defer(d3.json, "/rest/domain.geojson")
                .defer(d3.json, "/rest/event")
                .await(function (error, world, boundaries, evs) {
                    if (error) return $log.error(error);

                    land = topojson.feature(world, world.objects.land);
                    domain_boundaries = boundaries;
                    var circles = [];

                    var evs = evs["events"];

                    for (var i=0; i<evs.length; i++) {
                        var event = evs[i];

                        var circle = d3.geo.circle()
                            .angle(event["magnitude"] / 4)
                            .origin(function(x, y) {return [x, y]});
                        circles.push(circle(event["longitude"],
                                            event["latitude"]));
                    }
                    event_list = {type: "GeometryCollection",
                                  geometries: circles};
                    redraw();
                });

            function redraw() {
                context.clearRect(0, 0, width, height);

                context.lineWidth = .75 * pixelRatio;
                context.strokeStyle = "#000";
                context.fillStyle = "#eee";
                context.beginPath(), path(land), context.fill(), context.stroke();

                context.lineWidth = .5 * pixelRatio;
                context.strokeStyle = "#ccc";
                context.beginPath(), path(graticule), context.stroke();

                context.lineWidth = 2.0 * pixelRatio;
                context.strokeStyle = "#c00";
                context.beginPath(), path(domain_boundaries), context.stroke();

                context.lineWidth = 2.0 * pixelRatio;
                context.strokeStyle = "rgba(0, 0, 150, 1.0)";
                context.fillStyle = "rgba(0, 0, 150, 0.3)";
                context.beginPath(), path(event_list), context.fill(), context.stroke();
            };

            // Delay resize a bit as it is fairly expensive.
            d3.select(window).on('resize', resizeDelay);

            var delayIt;

            function resizeDelay() {
                clearTimeout(delayIt);
                delayIt = setTimeout(resize, 100);
            }

            function resize() {
                width = parseInt(d3.select(element[0]).style('width'));
                height = parseInt(d3.select(element[0]).style('height'));
                width *= pixelRatio;
                height *= pixelRatio;

                // update projection
                projection
                    .translate([width / 2, height / 2])
                    .clipExtent([
                        [dx, dx],
                        [width - dx, height - dx]
                    ]);

                // resize the map container
                canvas
                    .attr('width', width + 'px')
                    .attr('height', height + 'px');
                canvas
                    .style('width', width / pixelRatio + 'px')
                    .style('height', height / pixelRatio + 'px');

                redraw();
            }
        }
    };
});
