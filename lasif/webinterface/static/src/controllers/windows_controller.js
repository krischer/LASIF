var lasifApp = angular.module("LASIFApp");


lasifApp.controller('windowsController', function ($scope, $http) {

    $http.get("/rest/windows", {cache: true}).success(function(data) {
        $scope.windows = data;
    });

});

