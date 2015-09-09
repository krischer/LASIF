var lasifApp = angular.module("LASIFApp");


lasifApp.config(function ($routeSegmentProvider) {
    $routeSegmentProvider.

        when("/",            "home").
        when("/map/:event_name?", "map").
        when("/iterations",  "iterations").
        when("/windows/:iteration_name?/:event_name?",  "windows").
        when("/events",      "events").
        when("/events/list", "events.list").
        when("/events/depth_histogram", "events.depth_histogram").
        when("/events/time_histogram", "events.time_histogram").
        when("/events/magnitude_histogram", "events.magnitude_histogram").
        when("/events/magnitude_vs_depth", "events.magnitude_vs_depth").

        segment("home", {
            templateUrl: '/static/pages/home.html',
            controller: 'homeController'
        }).

        segment("map", {
            templateUrl: '/static/pages/map.html',
            controller: 'mapController'
        }).

        segment("iterations", {
            templateUrl: '/static/pages/iterations.html',
            controller: 'iterationsController'
        }).

        segment("windows", {
            templateUrl: '/static/pages/windows.html',
            controller: 'windowsController'
        }).

        segment("events", {
            templateUrl: '/static/pages/events.html',
            controller: 'eventsController'
        }).
        within().
            segment("list", {
                templateUrl: '/static/pages/events/list.html',
                controller: 'eventsController'
            }).
            segment("depth_histogram", {
                templateUrl: '/static/pages/events/depth_histogram.html',
                controller: 'eventsController'
            }).
            segment("magnitude_histogram", {
                templateUrl: '/static/pages/events/magnitude_histogram.html',
                controller: 'eventsController'
            }).
            segment("magnitude_vs_depth", {
                templateUrl: '/static/pages/events/magnitude_vs_depth.html',
                controller: 'eventsController'
            }).
            segment("time_histogram", {
                templateUrl: '/static/pages/events/time_histogram.html',
                controller: 'eventsController'
            });
});

