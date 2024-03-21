'use strict';

const $ = django.jQuery;

$(document).ready(function () {
    $('#birads_label').click(function () {
        const win = window.open(
            $(this).attr('href'),
            'BIRADS',
            'height=500,width=800,resizable=yes,scrollbars=yes'
        );
        win.focus();
        return false;
    })
})

