var lasifApp = angular.module("LASIFApp");


lasifApp.controller('waveformPlotController', function($scope, $log, $http) {

    $scope.tag_color_map = {}
    $scope.downloadInProgress = false;

    $http.get("/rest/available_data/" + $scope.$parent.event_name + "/"
        + $scope.$parent.station.station_name, {
        cache: true
    }).success(function(data) {
        var availableData = {
            "raw": _.map(data.raw, function(i) {
                $scope.tag_color_map[i] = "#ccc";
                return [i, true]
            }),
            "processed": _.map(data.processed, function(i, index) {
                $scope.tag_color_map[i] = colorbrewer["Greys"]["6"][(5 - index) % 6];
                return [i, false]
            }),
            "synthetic": _.map(data.synthetic, function(i, index) {
                $scope.tag_color_map[i] = colorbrewer["Reds"]["6"][(5 - index) % 6];
                return [i, false]
            })
        }
        $scope.availableData = availableData;
    });

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
                    .filter(function(j) {return j[1]})
                    .map(function(k) {return k[0];})
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
