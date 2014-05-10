var lasifApp = angular.module("LASIFApp");


lasifApp.controller('mapController', function ($scope, $log, $location, $routeSegment) {
    $scope.shownEvents = undefined;

    function getCurrentEvent() {
        var split_url = _($location.url().split("/")).compact().value();
        if (split_url[0] != "map") {
            return undefined
        }

        var event_name;
        if (split_url.length == 1) {
            event_name = undefined;
        }
        else {
            event_name = split_url[1];
        }
        return event_name;
    }

    // The currently shown events.
    $scope.shownEvents = getCurrentEvent();

    // Monitor the URL to be able to change the shown events.
    $scope.$on('$locationChangeSuccess', function () {
        $scope.shownEvents = getCurrentEvent();
    });
});
