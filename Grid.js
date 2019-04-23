"use strict";

class NoteWidget {

    constructor(index) {
        this.octave = 5;
        this.index = index;
        this.delta = -1;

        const octave_elem = $.parseHTML(`<p><div class="ui mini icon vertical buttons">
            <button class="ui button" data-value="6">+</button>
            <button class="ui button active" data-value="5">0</button>
            <button class="ui button" data-value="4">-</button>
        </div></p>`);

        const delta_elem = $.parseHTML(`<p><div class="ui icon top left pointing dropdown mini button">
            <div class="default text">&emptyset;</div>
            <div class="menu">
                <div class="item" data-value="-1">&emptyset;</div>
                <div class="item" data-value="0">1st</div>
                <div class="item" data-value="2">3rd</div>
                <div class="item" data-value="4">5th</div>
            </div>
        </div></p>`);

        const dual_button = $.parseHTML(`<div class="ui vertical buttons">
        <div class="ui icon top left pointing dropdown mini button">
            <div class="default text">&emptyset;</div>
            <div class="menu">
                <div class="item" data-value="-1">&emptyset;</div>
                <div class="item" data-value="0">1st</div>
                <div class="item" data-value="2">3rd</div>
                <div class="item" data-value="4">5th</div>
            </div>
        </div>
        <div class="ui icon top left pointing dropdown mini button">
            <div class="default text">0</div>
            <div class="menu">
                <div class="item" data-value="6">+</div>
                <div class="item" data-value="5">0</div>
                <div class="item" data-value="4">-</div>
            </div>
        </div>
        </div>
        `);

        const self = this;

        const octave_buttons = $(octave_elem).find('.ui.button');
        octave_buttons.click(function() {
            const current = $(this);
            const vv = parseFloat(current.attr('data-value'));
            octave_buttons.removeClass('active');
            current.addClass('active');
            self.octave = vv;
            self.update();
        });

        const delta_dropdown = $(dual_button).find('.ui.button');
        delta_dropdown.dropdown({
            on: 'hover',
            duration: 0,
            onChange: function() {
                self.delta = parseInt($(this).dropdown('get value'));
                self.update();
            },
        });

        // const elem = $.parseHTML('<div><p id="index"></p></div>');
        // elem.find('#index').text(`${index}`);
        const elem = $('<div>');
        //elem.append(delta_elem);
        //elem.append(octave_elem);
        elem.append(dual_button)

        this.elem = elem;
    }

    get note() {
        return this.delta < 0 ? null : create_note(0, this.delta, this.octave);
    }

    update() {
        // console.log('NoteWidget', 'update', this.index,     this.note);
        if (this.onUpdate) this.onUpdate();
    }
}
