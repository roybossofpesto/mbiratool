const wrap = (xx) => {
    return xx % 7;
};

const letters = "CDEFGAB";
const numbers = "1234567";
const tunings = [
    "Ionian",
    "Dorian",
    "Phyrgian",
    "Lydian",
    "Myxolydian",
    "Aeolian",
    "Locrian",
];
const song_names = []

const root_key_colors = ["#1c96fe", "#fe6e32", "#aee742", "#b75ac4", "#fbed00", "#d73535", "#ff5986"]
    .map((color) => {
        return chroma(color);
    });

$(document).ready(() => {
    const base_size = 360;
    const knob_size = 70;

    $('div.dial-old').each(function() {
        const paper = Raphael(this, base_size, base_size);
        const thickness = 4;
        const large_dot_radius = 10;
        const small_dot_radius = 5;
        const radius = base_size / 2 - thickness / 2 - large_dot_radius;
        paper.circle(base_size / 2, base_size / 2, radius).attr({
            'fill': 'transparent',
            'stroke-width': thickness,
            'stroke': 'black',
        });

        let hosho = 0;
        const dots = [];
        for (let kk = 0; kk < 48; kk++) {
            const is_large = (kk % 3 == hosho % 3);
            const color = root_key_colors[wrap(Math.floor(kk / 4))];
            const dot = paper
                .circle(base_size / 2, base_size / 2 - radius, is_large ? large_dot_radius : small_dot_radius)
                .rotate(360 * kk / 48, base_size / 2, base_size / 2)
                .attr({
                    'fill': color,
                    'stroke': 'black',
                    'stroke-width': thickness,
                })
            dots.push(dot);
        }

        $(this).click((evt) => {
            hosho += 1;
            dots.forEach((dot, kk) => {
                const is_large = (kk % 3 == hosho % 3);
                dot.animate({
                    'r': is_large ? large_dot_radius : small_dot_radius
                }, 100, '>')
            })
            evt.stopPropagation();
            evt.preventDefault();
        })
    })

    let tuning = parseInt($('#tuning-knob').val());
    let mode = parseInt($('#mode-knob').val());
    let use_letter_alphabet = $('#letters-checkbox').prop('checked');

    const update_dials = $('div.dial').map(function() {
        const paper = Raphael(this, base_size, base_size);
        paper.customAttributes.arc = function(centerX, centerY, startAngle, endAngle, innerR, outerR) {
            var radians = Math.PI / 180,
                largeArc = +(endAngle - startAngle > 180);
            // calculate the start and end points for both inner and outer edges of the arc segment
            // the -90s are about starting the angle measurement from the top get rid of these if this doesn't suit your needs
            outerX1 = centerX + outerR * Math.cos((startAngle - 90) * radians),
                outerY1 = centerY + outerR * Math.sin((startAngle - 90) * radians),
                outerX2 = centerX + outerR * Math.cos((endAngle - 90) * radians),
                outerY2 = centerY + outerR * Math.sin((endAngle - 90) * radians),
                innerX1 = centerX + innerR * Math.cos((endAngle - 90) * radians),
                innerY1 = centerY + innerR * Math.sin((endAngle - 90) * radians),
                innerX2 = centerX + innerR * Math.cos((startAngle - 90) * radians),
                innerY2 = centerY + innerR * Math.sin((startAngle - 90) * radians);

            // build the path array
            var path = [
                ["M", outerX1, outerY1], //move to the start point
                ["A", outerR, outerR, 0, largeArc, 1, outerX2, outerY2], //draw the outer edge of the arc
                ["L", innerX1, innerY1], //draw a line inwards to the start of the inner edge of the arc
                ["A", innerR, innerR, 0, largeArc, 0, innerX2, innerY2], //draw the inner arc
                ["z"] //close the path
            ];
            return {
                path: path
            };
        };
        const radius = base_size / 2;
        paper.circle(base_size / 2, base_size / 2, radius).attr({
            'fill': 'black',
            'stroke-width': 0,
            'stroke': '#f0f',
        });

        const sectors = []
        for (let kk = 0; kk < 48; kk++)
            sectors.push(paper
                .path()
                .attr({
                    "stroke-width": 0,
                    'stroke': "#f0f",
                    'fill': root_key_colors[Math.floor(kk / 4) % 7],
                    'arc': [base_size / 2, base_size / 2, 0, 360 / 48 + .5, 50, 160],
                })
                .rotate(360 * kk / 48, base_size / 2, base_size / 2))


        const update = () => {
            const chords = [];
            const value = mode + tuning;
            first_song_blocks.each(function(index) {
                const aa = wrap(value + 0 + (index > 2));
                const bb = wrap(value + 2 + (index > 1));
                const cc = wrap(value + 4 + (index > 0));
                chords.push(aa, aa, aa, aa, bb, bb, bb, bb, cc, cc, cc, cc)
            })
            console.log('update dial', chords, sectors)
            sectors.forEach((sector, index) => {
                sector.attr('fill', root_key_colors[chords[index]].brighten(index % 2 == 0 ? 0 : 1))
            })
        }
        return update;
    }).get();

    const update_mbiras = $('div.mbira').map(function() {
        const pen_width = 2;
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

        const reset_colors = () => {
            keys.forEach((foo) => {
                foo.attr('fill', root_key_colors[wrap(foo.key + tuning)])
            })
        };
        const keys = [];
        const create_key = (width, height, key) => {
            const key_third = (key + 2) % 7;
            const key_fifth = (key + 4) % 7;
            const path_string = "M0,0l0," + height + "c0," + width + "," + width + "," + width + "," + width + ",0l0," + (-height) + "z";
            const path_object = paper.path(path_string);
            path_object.attr({
                "stroke-width": 2 * pen_width,
                "fill": root_key_colors[wrap(key + tuning)],
            });
            path_object.key = key;
            keys.push(path_object);
            path_object.hover(() => {
                keys.forEach((foo, kk) => {
                    const base_color = root_key_colors[wrap(key + tuning)];
                    const color =
                        foo.key == key ? base_color :
                        foo.key == key_third ? base_color.brighten(2.5) :
                        foo.key == key_fifth ? base_color.brighten(1) :
                        "white";
                    foo.attr('fill', color);
                })
            }, reset_colors);
            return path_object;
        };

        const big_key_width = 24;
        const small_key_width = 15.35;
        const key_offset = 15;
        const lower_left_keys = [1, 6, 5, 4, 3, 2, 0];
        const upper_left_keys = [1, 0, 6, 5, 3, 4, 0];
        const right_keys = [0, 1, 2, 3, 4, 5, 6, 0, 1];
        lower_left_keys.forEach((key, kk) => {
            create_key(big_key_width, 210 + 10 * kk, key).translate(key_offset + big_key_width * (kk + .5), key_offset);
        })
        upper_left_keys.forEach((key, kk) => {
            create_key(big_key_width, 150 + 8 * kk, key).translate(key_offset + big_key_width * kk, key_offset);
        })
        create_key(big_key_width, 170, 2).translate(key_offset + big_key_width * upper_left_keys.length, key_offset);
        right_keys.forEach((key, kk) => {
            create_key(small_key_width, 150 - 5 * kk, key).translate(base_size - key_offset + small_key_width * (kk - right_keys.length), key_offset);
        })

        paper.rect(pen_width, 35, base_size - 2 * pen_width, 20, 0).attr({
            "stroke-width": 2 * pen_width,
            "fill": "#777"
        });

        return reset_colors;
    }).get();

    const first_song_blocks = $('#first-song>div>span');
    const second_song_blocks = $('#second-song>div>span');
    const third_song_blocks = $('#third-song>div>span');
    const update_songs = () => {
        const alphabet = use_letter_alphabet ? letters : numbers;
        const value = mode + use_letter_alphabet * tuning;
        const value_ = mode + tuning;
        first_song_blocks.each(function(index) {
            const aa = wrap(value + 0 + (index > 2));
            const bb = wrap(value + 2 + (index > 1));
            const cc = wrap(value + 4 + (index > 0));
            const aa_ = wrap(value_ + 0 + (index > 2));
            $(this)
                .text(`${alphabet[aa]}${alphabet[bb]}${alphabet[cc]}`)
                .css('background-color', root_key_colors[aa_]);
        })
        second_song_blocks.each(function(index) {
            const aa = wrap(value + 2 + (index < 2));
            const bb = wrap(value + 4 + (index != 2));
            const cc = wrap(value + 0 + (index == 0));
            const aa_ = wrap(value_ + 2 + (index < 2));
            $(this)
                .text(`${alphabet[aa]}${alphabet[bb]}${alphabet[cc]}`)
                .css('background-color', root_key_colors[aa_]);
        })
        third_song_blocks.each(function(index) {
            const aa = wrap(value + 4 + (index < 3));
            const bb = wrap(value + 0 + (index == 1));
            const cc = wrap(value + 2 + (index < 2));
            const aa_ = wrap(value_ + 4 + (index < 3));
            $(this)
                .text(`${alphabet[aa]}${alphabet[bb]}${alphabet[cc]}`)
                .css('background-color', root_key_colors[aa_]);
        })
    };

    const update_all = () => {
        update_songs();
        update_mbiras.forEach((foo) => foo());
        update_dials.forEach((foo) => foo());
    }

    // knob demo page http://anthonyterrien.com/demo/knob/

    $('#tuning-knob').knob({
        'width': knob_size,
        'height': knob_size,
        'min': 0,
        'max': 7,
        'displayInput': false,
        'cursor': 360 / 7,
        'fgColor': 'black',
        'thickness': .5,
        'format': (foo) => {
            tuning = foo
            $('div.view.tuning').text(tunings[wrap(tuning)]);
            update_all();
            return foo;
        },
    })
    $('#mode-knob').knob({
        'width': knob_size,
        'height': knob_size,
        'min': 0,
        'max': 7,
        'displayInput': false,
        'cursor': 360 / 7,
        'fgColor': 'black',
        'thickness': .5,
        'format': (foo) => {
            mode = foo
            $('div.view.mode').text(numbers[wrap(mode)]);
            update_all();
            return foo;
        },
    })
    $('#letters-checkbox').on('change', function() {
        use_letter_alphabet = this.checked;
        update_all();
    });

})
