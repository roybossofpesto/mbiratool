"use strict";

class NoteWidget {

    constructor() {
        this.__chord = null;
        this.__octave = 5;
        this.__delta = -1;

        const dual_button = $.parseHTML(`
        <div class="ui vertical fluid icon buttons">
            <div class="ui icon top left pointing dropdown mini black button delta">
                <div class="default text">&emptyset;</div>
                <div class="menu">
                    <div class="item" data-value="-1">&emptyset;</div>
                    <div class="item" data-value="0">1st</div>
                    <div class="item" data-value="2">3rd</div>
                    <div class="item" data-value="4">5th</div>
                </div>
            </div>
            <div class="ui icon top left pointing dropdown mini black button octave">
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

        this.__octave_dropdown = $(dual_button).find('.ui.dropdown.octave');
        this.__octave_dropdown.dropdown({
            on: 'hover',
            duration: 0,
            onChange: function() {
                self.__octave = parseInt($(this).dropdown('get value'));
                self.update();
            },
        });

        this.__delta_dropdown = $(dual_button).find('.ui.dropdown.delta');
        this.__delta_dropdown.dropdown({
            on: 'hover',
            duration: 0,
            onChange: function() {
                self.__delta = parseInt($(this).dropdown('get value'));
                self.update();
            },
        });

        this.elem = dual_button;
    }

    get delta() {
        return this.__delta;
    }

    set delta(value) {
        this.__delta = value;
        this.__delta_dropdown.dropdown('set selected', value.toString());
    }

    get octave() {
        return this.__octave;
    }

    set octave(value) {
        this.__octave = value;
        this.__octave_dropdown.dropdown('set selected', value.toString());
    }

    get chord() {
        return this.__chord;
    }

    set chord(value) {
        this.__chord = wrap(value);
        this.update();
    }

    get note() {
        return create_note(this.__chord, this.__delta, this.__octave);
    }

    update() {
        // console.log('NoteWidget', 'update', this.delta, this.octave, this.index, this.note);
        const chord_color = chord_colors[this.__chord];
        this.__octave_dropdown.css('background-color', chord_color);
        this.__delta_dropdown.css('background-color', this.note == null ? "#000" : chord_color)
        if (this.onUpdate) this.onUpdate();
    }
}
