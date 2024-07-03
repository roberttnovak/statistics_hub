$(document).ready(function() {
    $("#help-button").click(function() {
        var explanationSection = $("#explanation-preprocessing-section");
        if (explanationSection.is(":visible")) {
            explanationSection.hide();
        } else {
            explanationSection.show();
        }
    });
});