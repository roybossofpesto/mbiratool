"use strict";

const root_key_colors = ["#1c96fe", "#feb831", "#aee742", "#b75ac4", "#15cdc2", "#fa2424", "#ff5986"].map(color => chroma(color));

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

        this.octave_dropdown = $(dual_button).find('.ui.dropdown.octave');
        this.octave_dropdown.dropdown({
            on: 'hover',
            duration: 0,
            onChange: function() {
                self.__octave = parseInt($(this).dropdown('get value'));
                self.update();
            },
        });

        this.delta_dropdown = $(dual_button).find('.ui.dropdown.delta');
        this.delta_dropdown.dropdown({
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
        this.delta_dropdown.dropdown('set selected', value);
        this.update();
    }

    get octave() {
        return this.__octave;
    }

    set octave(value) {
        this.__octave = value;
        this.octave_dropdown.dropdown('set selected', value);
        this.update();
    }

    get chord() {
        return this.__chord;
    }

    set chord(root_key) {
        this.__chord = root_key % 7;
        this.update();
    }

    get note() {
        return this.delta < 0 ? null : create_note(this.chord, this.delta, this.octave);
    }

    update() {
        // console.log('NoteWidget', 'update', this.delta, this.octave, this.index, this.note);
        const root_color = root_key_colors[this.__chord];
        this.delta_dropdown.css('background-color', this.note == null ? "black" : root_color)
        this.octave_dropdown.css('background-color', root_color);
        if (this.onUpdate) this.onUpdate();
    }
}
