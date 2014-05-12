var lasifApp = angular.module("LASIFApp");


lasifApp.controller('waveformPlotController', function ($scope, $log, $http) {

    $http.get("/rest/available_data/" + $scope.$parent.event_name + "/"
        + $scope.$parent.station.station_name, {
        cache: true
    }).success(function (data) {
        var availableData = {
            "raw": true,
            "processed": _.map(data.processed, function (i) {
                return [i, false]
            }),
            "synthetic": _.map(data.synthetic, function (i) {
                return [i, false]
            })
        }
        $scope.availableData = availableData;
    });

    $scope.zData = [];
    $scope.nData = [];
    $scope.eData = [];

    $scope.colorFunction = function () {
        return function (d, i) {
            if (d.key == "raw") {
                return "#CCC";
            }
            else {
                return "#F00";
            }
        }
    };

    $scope.$watch("availableData", function (new_value, old_value) {
        if (new_value == old_value) {
            return
        }
        // Figure out what changed.
        if (!old_value || new_value.raw != old_value.raw) {
            // Remove if deselected.
            if (new_value.raw == false) {
                _.forEach(["z", "n", "e"], function (i) {
                    $scope[i + "Data"] = _.filter($scope[i + "Data"],
                        function (i) {
                            if (i.key == "raw") {
                                return false
                            }
                            else {
                                return true
                            }
                        });
                });
            }
            // Otherwise add.
            else {
                $http.get("/rest/get_data/" + $scope.$parent.event_name + "/"
                    + $scope.$parent.station.station_name + "/" + "raw", {
                    cache: false
                }).success(function (data) {
                    _.forEach(["Z", "N", "E"], function (i) {
                        if (data[i]) {
                            var cur_data = _.clone(
                                $scope[i.toLowerCase() + "Data"]);
                            cur_data.push({
                                key: "raw",
                                values: data[i]
                            });
                            $scope[i.toLowerCase() + "Data"] = cur_data;
                        }
                    });
                });
            }
        }
        $log.log("Old:", old_value);
        $log.log("New:", new_value);
    }, true);
});
