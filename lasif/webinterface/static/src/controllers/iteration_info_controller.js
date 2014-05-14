var lasifApp = angular.module("LASIFApp");


lasifApp.controller('iterationInfoController', function($scope, $http) {
    $scope.stf = [];

    $http.get("/rest/iteration/" + $scope.iteration_name, {cache: true })
        .success(function(data) {
            $scope.info = data;
            $scope.stf = [
                {
                    key: "STF",
                    values: _.map(data.source_time_function.data, function(i, j) {
                        return [j * data.source_time_function.delta, i];
                    })
                }
            ];
        });

    $scope.pretty_solver_settings = function() {
        if (!$scope.info) {
            return "";
        }
        else {
            return JSON.stringify($scope.info.solver_settings, null, 2);
        }
    };

    $scope.yAxisTickFormatFunction = function() {
        return function(d) {
            return "";
        }
    };
});
