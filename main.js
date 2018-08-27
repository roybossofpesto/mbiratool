const wrap = (xx, alphabet) => {
    return alphabet[xx % 7];
};

const letters = "ABCDEFG";
const numbers = "1234567";
const tunings = [
    "Ionian",
    "Dorian",
    "Phyrgian",
    "Lydian",
    "Myxolydian",
    "Aeolian",
    "Locrian",
]

$(document).ready(() => {
    const base_size = 360;
    $('div.dial').each(function() {
        const paper = Raphael(this, base_size, base_size);
        paper.circle(base_size / 2, base_size / 2, base_size / 2 - 10).attr({
            'fill': '#f0f',
            'stroke-width': 20,
            'stroke': '#0ff',
        });
    })

    $('div.mbira').each(function() {
        const pen_width = 2;
        const keys = [0, 1, 2, 3, 4, 5, 6];
        const paper = Raphael(this, base_size, base_size);

        paper.rect(pen_width, pen_width, base_size - 2 * pen_width, base_size - 2 * pen_width, 10).attr({
            "stroke-width": 2 * pen_width,
            "fill": "#eee"
        });
        paper.rect(20, 40, base_size - 40, base_size - 80, 0).attr({
            "stroke-width": 2 * pen_width,
            "fill": "#777"
        });
        paper.circle(base_size - 55, base_size - 75, 20).attr({
            "stroke-width": 2 * pen_width,
            "fill": "#fff"
        });

        const key_shape = (width, height, key) => {
            const key_third = (key + 2) % keys.length;
            const key_fifth = (key + 4) % keys.length;
            const key_colors = ["#56b0fd", "#fe6e32", "#aee742", "#b75ac4", "#fff436", "#40534f", "#ffc0d1"];
            const path_string = "M0,0l0," + height + "c0," + width + "," + width + "," + width + "," + width + ",0l0," + (-height) + "z";
            const path_object = paper.path(path_string);
            path_object.attr({
                "stroke-width": 2 * pen_width,
                "fill": key_colors[key],
            });
            path_object.key = key;
            path_object.hover(function() {
                console.log('in', this.key + 1)
                // for (let kk = 0; kk < keys.length; kk++) {
                //     if (kk == key) continue;
                //     if (kk == key_third) continue;
                //     if (kk == key_fifth) continue;
                //     for (var ll = 0; ll < keys[kk].length; ll++) keys[kk][ll].attr("fill", "#fff");
                // }
                // document.body.style.cursor = "pointer";
            }, function() {
                console.log('out', this.key + 1)
                // document.body.style.cursor = "auto";
                // for (let kk = 0; kk < keys.length; kk++) {
                //     for (var ll = 0; ll < keys[kk].length; ll++) keys[kk][ll].attr("fill", key_colors[kk]);
                // }
            });
            return path_object;
        };

        const big_key_width = 24;
        const small_key_width = 15.35;
        const key_offset = 15;
        const lower_left_keys = [1, 6, 5, 4, 3, 2, 0];
        const upper_left_keys = [1, 0, 6, 5, 3, 4, 0];
        const right_keys = [0, 1, 2, 3, 4, 5, 6, 0, 1];
        lower_left_keys.forEach((key, kk) => {
            key_shape(big_key_width, 210 + 10 * kk, key).translate(key_offset + big_key_width * (kk + .5), key_offset);
        })
        upper_left_keys.forEach((key, kk) => {
            key_shape(big_key_width, 150 + 8 * kk, key).translate(key_offset + big_key_width * kk, key_offset);
        })
        key_shape(big_key_width, 170, 2).translate(key_offset + big_key_width * upper_left_keys.length, key_offset);
        right_keys.forEach((key, kk) => {
            key_shape(small_key_width, 150 - 5 * kk, key).translate(base_size - key_offset + small_key_width * (kk - right_keys.length), key_offset);
        })

        paper.rect(pen_width, 35, base_size - 2 * pen_width, 20, 0).attr({
            "stroke-width": 2 * pen_width,
            "fill": "#777"
        });

    })

    const first_song_blocks = $('#first-song>div>span');
    const second_song_blocks = $('#second-song>div>span');
    const third_song_blocks = $('#third-song>div>span');
    const update_songs = (value, alphabet) => {
        first_song_blocks.each(function(index) {
            const aa = wrap(value + 0 + (index > 2), alphabet);
            const bb = wrap(value + 2 + (index > 1), alphabet);
            const cc = wrap(value + 4 + (index > 0), alphabet);
            $(this).text(`${aa}${bb}${cc}`)
        })
        second_song_blocks.each(function(index) {
            const aa = wrap(value + 2 + (index < 2), alphabet);
            const bb = wrap(value + 4 + (index != 2), alphabet);
            const cc = wrap(value + 0 + (index == 0), alphabet);
            $(this).text(`${aa}${bb}${cc}`)
        })
        third_song_blocks.each(function(index) {
            const aa = wrap(value + 4 + (index < 3), alphabet);
            const bb = wrap(value + 0 + (index == 1), alphabet);
            const cc = wrap(value + 2 + (index < 2), alphabet);
            $(this).text(`${aa}${bb}${cc}`)
        })
    };

    let current_tuning = parseInt($('#tuning-slider').val());
    let current_mode = parseInt($('#mode-slider').val());
    let use_letter_alphabet = $('#letters-checkbox').prop('checked');

    // knob demo page http://anthonyterrien.com/demo/knob/
    $('#tuning-slider').knob({
        'width': 100,
        'height': 100,
        'min': 0,
        'max': 7,
        'displayInput': false,
        'cursor': 52,
        'fgColor': 'black',
        'format': function(foo) {
            current_tuning = foo
            $('div.view.tuning').text(wrap(current_tuning, tunings));
            update_songs(current_mode + current_tuning * use_letter_alphabet, use_letter_alphabet ? letters : numbers);
            return foo;
        },
    })
    $('#mode-slider').knob({
        'width': 100,
        'height': 100,
        'min': 0,
        'max': 7,
        'displayInput': false,
        'cursor': 52,
        'fgColor': 'black',
        'format': function(foo) {
            current_mode = foo
            $('div.view.mode').text(wrap(current_mode, numbers));
            update_songs(current_mode + current_tuning * use_letter_alphabet, use_letter_alphabet ? letters : numbers);
            return foo;
        },
    })
    $('#letters-checkbox').on('change', function() {
        use_letter_alphabet = this.checked;
        update_songs(current_mode + current_tuning * use_letter_alphabet, use_letter_alphabet ? letters : numbers);
    });

})
