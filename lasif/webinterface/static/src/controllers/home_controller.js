var lasifApp = angular.module("LASIFApp");

lasifApp.controller('homeController', function ($scope, latestOutput) {

    latestOutput.success(function(data) {
        $scope.latest_output = data.folders;
    });

});
