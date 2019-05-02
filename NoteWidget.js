"use strict";

class NoteWidget {

    constructor() {
        this.__chord = null;
        this.__octave = 5;
        this.__delta = -1;
        this.__enabled = true;

        const dual_button = $($.parseHTML(`
        <div class="ui column">
            <div class="ui vertical fluid icon buttons">
                <div class="ui icon top left pointing dropdown mini black button delta">
                    <div class="default text">&emptyset;</div>
                    <div class="menu">
                        <div class="item" data-value="-1">&emptyset;</div>
                        <div class="item" data-value="0">1st</div>
                        <div class="item" data-value="4">5th</div>
                        <div class="item" data-value="2">3rd</div>
                        <div class="item" data-value="3">4th</div>
                    </div>
                </div>
                <div class="ui icon top left pointing dropdown mini black button octave">
                    <div class="default text">&#x2582;</div>
                    <div class="menu">
                        <div class="item" data-value="6">&#x2588;</div>
                        <div class="item" data-value="5">&#x2585;</div>
                        <div class="item" data-value="4">&#x2582;</div>
                    </div>
                </div>
                <div class="ui fluid icon button gate"></div>
            </div>
        </div>
        `));

        const self = this;

        this.__enabled_button = dual_button.find('.ui.button.gate');
        this.__enabled_button.click(() => {
            self.__enabled = !self.__enabled;
            self.update();
        })

        this.__octave_dropdown = dual_button.find('.ui.dropdown.octave');
        this.__octave_dropdown.dropdown({
            on: 'hover',
            duration: 0,
            onChange: function() {
                self.__octave = parseInt($(this).dropdown('get value'));
                self.update();
            },
        });

        this.__delta_dropdown = dual_button.find('.ui.dropdown.delta');
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
        this.__delta_dropdown.dropdown('set selected', this.__delta.toString());
    }

    get octave() {
        return this.__octave;
    }

    set octave(value) {
        this.__octave = value;
        this.__octave_dropdown.dropdown('set selected', this.__octave.toString());
    }

    get chord() {
        return this.__chord;
    }

    set chord(value) {
        this.__chord = wrap(value);
        this.update();
    }

    get enabled() {
        return this.__enabled;
    }

    set enabled(value) {
        this.__enabled = (value == true);
        this.update();
    }

    get note() {
        return create_note(this.__chord, this.__delta, this.__octave);
    }

    set note(value) {
        if (value) {
            this.__chord = wrap(value.chord);
            this.__delta = value.delta;
            this.__octave = value.octave;
            this.__delta_dropdown.dropdown('set selected', this.__delta.toString());
            this.__octave_dropdown.dropdown('set selected', this.__octave.toString());
        } else {
            this.__delta = -1;
            this.__delta_dropdown.dropdown('set selected', this.__delta.toString());
        }
    }

    update() {
        // console.log('NoteWidget', 'update', this.delta, this.octave, this.index, this.note);
        const octave_backcolor = chord_colors[this.__chord];
        this.__octave_dropdown.css('background-color', octave_backcolor.css());

        const delta_backcolor = delta_brighten(octave_backcolor, this.__delta);
        const delta_color = this.__delta <= 0 || this.__delta == 4 ? chroma("white") : chroma.mix(octave_backcolor, "black");
        this.__delta_dropdown.css('background-color', delta_backcolor.css());
        this.__delta_dropdown.css('color', delta_color.css())

        this.__enabled_button.css('background-color', this.__enabled ? octave_backcolor.css() : "black");

        if (this.onUpdate) this.onUpdate(this.note, this.enabled);
    }

    ping(duration = 300) {
        this.elem.find('.icon.buttons')
            .css('margin-right', '10px')
            .animate({
                marginRight: 0
            }, duration);
    }
}
