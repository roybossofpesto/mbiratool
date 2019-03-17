"use strict";

class NoteWidget {

    constructor(index) {
        this.octave = 0;
        this.index = index;

        // const foo =
        //     '<p id="lol"><div class="ui icon top left pointing gap dropdown mini button"> <div class="default text">1st</div> <div class="menu"> <div class="item" data-value="0">1st</div> <div class="item" data-value="2">3rd</div> <div class="item" data-value="4">5th</div> </div> </div> </p> ';
        //
        // const elem = $($.parseHTML(foo));
        const octave_elem = $.parseHTML(`<div class="ui compact icon vertical octave buttons">
            <button class="ui button" data-value="1">+</button>
            <button class="ui button active" data-value="0">0</button>
            <button class="ui button" data-value="-1">-</button>
            </div>`);
        const elem = $.parseHTML('<p><i class="ui warning icon"></i> <span id="index">aa</span></p>');
        // elem.children(0).append(aa);
        $(elem).find('#index').text(`${index}`);
        $(elem).append(octave_elem);

        const self = this;

        const octave_buttons = $(octave_elem).find('button');
        octave_buttons.click(function() {
            const current = $(this);
            const vv = parseFloat(current.attr('data-value'));
            octave_buttons.removeClass('active');
            current.addClass('active');
            self.octave = vv;
            self.update();
        });

        this.elem = elem;
    }

    get note() {

    }

    update() {
        console.log('NoteWidget', 'update', this.index, this.octave);
    }
}
