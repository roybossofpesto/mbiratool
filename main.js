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
    $('div.dial').each(function() {
        const paper = Raphael(this, 400, 400);
        paper.circle(200, 200, 150).attr({
            'fill': '#f0f',
            'stroke-width': 20,
            'stroke': '#0ff',
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

    let current_tuning = 0;
    let current_mode = 0;
    let use_letter_alphabet = false;
    $('#tuning-slider').on('input', function() {
        current_tuning = parseInt($(this).val());
        $('div.tuning-view').text(wrap(current_tuning, tunings));
        update_songs(current_mode + current_tuning * use_letter_alphabet, use_letter_alphabet ? letters : numbers);
    })
    $('#mode-slider').on('input', function() {
        current_mode = parseInt($(this).val());
        $('div.mode-view').text(wrap(current_mode, numbers));
        update_songs(current_mode + current_tuning * use_letter_alphabet, use_letter_alphabet ? letters : numbers);
    })
    $('#letters-checkbox').on('change', function() {
        use_letter_alphabet = this.checked;
        update_songs(current_mode + current_tuning * use_letter_alphabet, use_letter_alphabet ? letters : numbers);
    });

})
