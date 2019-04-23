"use strict";

const root_key_colors = ["blue", "orange", "green", "purple", "teal", "red", "pink"];

class NoteWidget {

    constructor(index) {
        this.__chord = null;
        this.octave = 5;
        this.index = index;
        this.delta = -1;
        this.chord_color = "black";
        this.root_key_color = "black";

        /*
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
                */

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

        const octave_dropdown = $(dual_button).find('.ui.dropdown.octave');
        octave_dropdown.dropdown({
            on: 'hover',
            duration: 0,
            onChange: function() {
                self.octave = parseInt($(this).dropdown('get value'));
                self.update();
            },
        });

        const delta_dropdown = $(dual_button).find('.ui.dropdown.delta');
        delta_dropdown.dropdown({
            on: 'hover',
            duration: 0,
            onChange: function() {
                const current = $(this);
                const vv = parseInt(current.dropdown('get value'));

                delta_dropdown.removeClass('active');
                current.addClass('active');

                self.delta = vv;

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
        return this.delta < 0 ? null : create_note(this.chord, this.delta, this.octave);
    }

    set chord(root_key) {
        this.__chord = root_key % 7;
        this.update();
    }

    get chord() {
        return this.__chord;
    }

    update() {
        // console.log('NoteWidget', 'update', this.delta, this.octave, this.index, this.note);
        const root_color = root_key_colors[this.__chord];

        const delta_dropdown = this.elem.find('.ui.dropdown.delta');
        const octave_dropdown = this.elem.find('.ui.dropdown.octave');

        delta_dropdown.removeClass(this.chord_color);
        this.chord_color = this.note == null ? "black" : root_color;
        delta_dropdown.addClass(this.chord_color);

        octave_dropdown.removeClass(this.root_key_color);
        this.root_key_color = root_color;
        octave_dropdown.addClass(this.root_key_color);

        if (this.onUpdate) this.onUpdate();
    }
}
