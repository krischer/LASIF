var lasifApp = angular.module("LASIFApp");

/* Filter truncating a string. Based on
/* http://igreulich.github.io/angular-truncate */
lasifApp.filter('truncate', function () {
    return function (text, length, end) {
        if (text != undefined) {
            if (isNaN(length)) {
                length = 25;
            }

            if (end === undefined) {
                end = "...";
            }

            if (text.length <= length || text.length - end.length <= length) {
                return text;
            } else {
                return String(text).substring(0, length - end.length) + end;
            }
        }
    };
});


// Format UTC dates as JS by default assumes local timezones.
lasifApp.filter('UTCDateTime', function () {
    return function(input) {
        var dt = new Date(input);
        return dt.toISOString();
    }
});


// Formats a Datestring to human readable time ago
// Uses moment.js.
lasifApp.filter('fromNow', function () {
  return function(date) {
    return moment(date).fromNow();
  }
});


// Converts "aa_bb_cc" to "AA BB CC"
lasifApp.filter('removeUnderscoresUppercase', function () {
    return function(input) {
        return input.replace("_", " ").toUpperCase();
    }
});
