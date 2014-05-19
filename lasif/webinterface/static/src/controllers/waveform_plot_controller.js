var lasifApp = angular.module("LASIFApp");


lasifApp.controller('waveformPlotController', function($scope, $log, $http) {

    $scope.tag_color_map = {}
    $scope.downloadInProgress = false;

    $http.get("/rest/available_data/" + $scope.$parent.event_name + "/"
        + $scope.$parent.station.station_name, {
        cache: true
    }).success(function(data) {
        var availableData = {
            "raw": _.map(data.raw, function(components, tag) {
                $scope.tag_color_map[tag] = "#ccc";
                return {
                    tag: tag,
                    show: true,
                    components: components
                };
            })
        };

        var index = 0;
        availableData.processed = _.map(data.processed, function(components, tag) {
            $scope.tag_color_map[tag] = colorbrewer["Greys"]["6"][(5 - index++) % 6];
            return {
                tag: tag,
                show: false,
                components: components
            };
        });

        index = 0;
        availableData.synthetic = _.map(data.synthetic, function(components, tag) {
            $scope.tag_color_map[tag] = colorbrewer["Reds"]["5"][(4 - index++) % 5];
            return {
                tag: tag,
                show: false,
                components: components
            };
        });
        $log.log(availableData);
        $log.log($scope.tag_color_map);
        $scope.availableData = availableData;
    });

    // Function returning the smallest and largest time value across all
    // arrays. This will make sure the plots share the xAxis.
//    $scope.forceX =
//        $log.log("========")
//        $log.log(x);
//        $log.log("========")
//    }

    // The data that will actually be plotted. Assign anew to trigger a redraw!
    $scope.dataZ = [];
    $scope.dataE = [];
    $scope.dataN = [];

    $scope.colorFunction = function() {
        return function(d, i) {
            return $scope.tag_color_map[d.key];
        }
    };

    $scope.xAxisTickFormatFunction = function() {
        return function(d) {
            return d3.time.format.utc('%H:%M:%S')(new Date(d * 1000));
        }
    }

    $scope.yAxisTickFormatFunction = function() {
        return function(d) {
            return ""
        }
    }

    $scope.$watch("availableData", function(newV, oldV) {
        if (newV == oldV) {
            return
        }

        $scope.downloadInProgress = true;

        var should_be_plotted = _(newV)
            .map(function(i) {
                return _(i)
                    .filter(function(j) {return j.show})
                    .map(function(k) {return k.tag;})
                    .value();
            })
            .flatten()
            .value();

        var tempDataScopes = {};
        // Remove unnecessarily plotted values.
        _.forEach(["Z", "N", "E"], function(component) {
            tempDataScopes[component] = _.filter($scope["data" + component],
                function(i) {return _.contains(should_be_plotted, i.key)});
        });

        var data_scopes = [$scope.dataZ, $scope.dataN, $scope.dataE]
        var is_plotted = _(tempDataScopes)
            .map(function(i) {return i})
            .flatten()
            .map("key")
            .union()
            .value()

        function applyScopes() {
            _.forEach(tempDataScopes, function(value, key) {
                $scope["data" + key] = value
            });
            $scope.downloadInProgress = false;
        };

        var needs_plotting = _.difference(should_be_plotted, is_plotted);
        if (!needs_plotting || !needs_plotting[0]) {
            applyScopes();
            return
        }

        if (needs_plotting.length > 1) {
            $log.error("Cannot download two tags simultaneously");
        }

        needs_plotting = needs_plotting[0];

        $http.get("/rest/get_data/" + $scope.$parent.event_name + "/"
            + $scope.$parent.station.station_name + "/" + needs_plotting, {
            cache: false
        }).success(function(data) {
            _.forEach(["Z", "N", "E"], function(i) {
                if (data[i]) {
                    tempDataScopes[i].push({
                        key: needs_plotting,
                        values: data[i]
                    });
                }
            });
            applyScopes();
        });
    }, true);
})
;
