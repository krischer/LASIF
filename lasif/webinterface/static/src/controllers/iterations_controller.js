var lasifApp = angular.module("LASIFApp");


lasifApp.controller('iterationsController', function ($scope, $http, $log, $modal) {

    $http.get("/rest/iteration", {cache: true}).success(function(data) {
        $scope.iterations = data.iterations;
    })

    $scope.showIteration = function(iteration_name) {
        var modal = $modal({
            title: "Iteration " + iteration_name,
            template: "/static/templates/iteration_info.tpl.html",
            persist: false,
            show: true});
        // Set scope of modal.
        modal.$scope.iteration_name = iteration_name;
    }
});

