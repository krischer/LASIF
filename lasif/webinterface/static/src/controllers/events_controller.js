var lasifApp = angular.module("LASIFApp");


lasifApp.controller('eventsController', function ($scope, eventList, $routeSegment, $log) {
    $scope.$routeSegment = $routeSegment;

    $scope.orderByField = 'event_name';
    $scope.reverseSort = false;

    eventList.success(function(data) {
        $scope.events = data.events;

        // Prepare the histogram data.
        var depth_data = [];
        var time_data = [];
        var magnitude_data = [];
        for (var key in $scope.events) {
            var event = $scope.events[key];
            depth_data.push(event.depth_in_km);
            time_data.push(event.origin_time);
            magnitude_data.push(event.magnitude);
        }


        // Data for Depth Histogram
        // Always go from 0 to 700 km.
        var depth_ticks = d3.scale.linear()
            .domain([0, 700])
            .range([0, 700])
            .ticks(20);

        var depth_bins = d3.layout.histogram()
            .bins(depth_ticks)(depth_data);

        var depth_values = [];

        for (var i=0; i < depth_bins.length; i++) {
            depth_values.push([
                depth_ticks[i] + (depth_ticks[i + 1] - depth_ticks[i]) * 0.5,
                depth_bins[i].length
            ]);
        }

        $scope.depthXAxisTickFormatFunction = function(){
            var half_range = (depth_ticks[1] - depth_ticks[0]) / 2.0;
            return function(d){
                return (d - half_range) + "-" + (d  + half_range)+ " km";
            };
        };

        $scope.depthHistogramData = [{
            "key": "Depth in km",
            "values": depth_values
        }];

        $scope.depthToolTipContentFunction = function(){
            return function(key, x, y, e, graph) {
                return parseInt(y) + " events at a depth range of " + x;
            }
        };


        var magnitude_ticks = d3.scale.linear()
            .domain([_.min(magnitude_data), _.max(magnitude_data)])
            .range([_.min(magnitude_data), _.max(magnitude_data)])
            .ticks(20);

        var magnitude_bins = d3.layout.histogram()
            .bins(magnitude_ticks)(magnitude_data);

        var magnitude_values = [];

        for (var i=0; i < magnitude_bins.length; i++) {
            magnitude_values.push([
                magnitude_ticks[i] +
                    (magnitude_ticks[i + 1] - magnitude_ticks[i]) * 0.5,
                magnitude_bins[i].length
            ]);
        }

        var format = d3.format('i');
        $scope.formatInt = function(){
            return function(d) {
                return format(d);
            }
        };

        $scope.magnitudeXAxisTickFormatFunction = function(){
            return function(d){
                return d.toFixed(2)
            };
        };

        $scope.magnitudeHistogramData = [{
            "key": "Magnitude",
            "values": magnitude_values
        }];

        $scope.magnitudeToolTipContentFunction = function(){
            var half_range = (magnitude_ticks[1] - magnitude_ticks[0]) / 2.0;
            return function(key, x, y) {
                return parseInt(y) + " event in a magnitude range of " +
                    (parseFloat(x) - half_range).toFixed(2) + "-" +
                    (parseFloat(x)  + half_range).toFixed(2);
            }
        };

        var data_vs_mag = [];

        for (var i=0; i < depth_data.length; i++) {
            data_vs_mag.push({"x": depth_data[i], "y": magnitude_data[i], "size": magnitude_data[i]});
        }

        $scope.magnitudeVsDepthData = [
            {"key": "Magnitude vs Depth",
             "values": data_vs_mag}
        ];

        $scope.magnitudeVsDepthXAxisTickFormatFunction = function(){
            return function(d){
                return d + " km";
            };
        };

        // Create counts per year.
        var year_bins = _.countBy(time_data, function(dt) {
            return new Date(dt).getFullYear();
        });
        // Make pairs
        year_bins = _.pairs(year_bins);
        _.each(year_bins, function(i) {i[0] = parseInt(i[0])});

        // Fill missing values.
        var desired_years = _.range(
            _.min(year_bins, function(i) {return i[0]})[0],
            _.max(year_bins, function(i) {return i[0]})[0] + 1);

        var available_years = _.map(year_bins, function(i) {return i[0]});

        _.each(_.difference(desired_years, available_years), function(i) {
            year_bins.unshift([i, 0]);
        });

        year_bins = _.sortBy(year_bins, function(i) {return i[0]});

        $scope.timeHistogramData = [{
            "key": "Time",
            "values": year_bins
        }];

        $scope.timeToolTipContentFunction = function(){
            return function(key, x, y) {
                return parseInt(y) + " event in " + x + ".";
            }
        };

    });

});
