var lasifApp = angular.module("LASIFApp");


lasifApp.controller('windowsController', function($scope, $log, $http, $location) {

    $http.get("/rest/windows", {cache: true}).success(function(data) {
        $scope.windows = data;
        $scope.iteration_dropdown = _.map(_.keys(data), function(value) {
            return {
                'text': 'Iteration ' + value,
                'href': '#/windows/' + value
            }
        });
    });

    function parseURL() {
        // The route segement provider does not appear to always be up to
        // date so we manually parse the URL here.
        var split_url = _($location.url().split("/")).compact().value();
        $scope.current_iteration = split_url[1];
        $scope.current_event = split_url[2];
    }

    parseURL();
    $scope.$on('$locationChangeSuccess', function() {
        parseURL()
        if ($scope.current_iteration) {
            $scope.window_distance_plot_dropdown = _.map(
                $scope.windows[$scope.current_iteration], function(value) {
                    return {
                        'text': value,
                        'href': '#/windows/' + $scope.current_iteration + '/' + value
                    }
                })
        }
        else {
            $scope.window_distance_plot_dropdown = [];
        }
    });

});

