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
                        <div class="item" data-value="5">6th</div>
                    </div>
                </div>
                <div class="ui icon top left pointing dropdown mini black button octave">
                    <div class="default text">&#x2585;</div>
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
        });

        this.__octave_dropdown = dual_button.find('.ui.dropdown.octave');
        this.__octave_dropdown.dropdown({
            on: 'hover',
            duration: 0,
            onChange: function() {
                const value = parseInt($(this).dropdown('get value'));
                if (value == self.__octave) return;
                self.__octave = value;
                self.update();
            },
        });

        this.__delta_dropdown = dual_button.find('.ui.dropdown.delta');
        this.__delta_dropdown.dropdown({
            on: 'hover',
            duration: 0,
            onChange: function() {
                const value = parseInt($(this).dropdown('get value'));
                if (value == self.__delta) return;
                self.__delta = value;
                self.update();
            },
        });

        this.elem = dual_button;
    }

    get delta() {
        return this.__delta;
    }

    set delta(value) {
        if (value == undefined) return;
        if (value == this.__delta) return;
        this.__delta_dropdown.dropdown('set selected', value.toString());
    }

    get octave() {
        return this.__octave;
    }

    set octave(value) {
        if (value == undefined) return;
        if (value == this.__octave) return;
        this.__octave_dropdown.dropdown('set selected', value.toString());
    }

    get chord() {
        return this.__chord;
    }

    set chord(value) {
        value = wrap(value);
        if (value == this.__chord) return;
        this.__chord = value;
        this.update();
    }

    get enabled() {
        return this.__enabled;
    }

    set enabled(value) {
        value = (value == true);
        if (value == this.__enabled) return;
        this.__enabled = value;
        this.update();
    }

    get note() {
        return create_note(this.__chord, this.__delta, this.__octave);
    }

    set note(value) {
        if (value != null && value.chord != undefined && value.delta != undefined && value.octave != undefined) {
            value.chord = wrap(value.chord);
            const should_update =
                value.chord != this.__chord ||
                value.delta != this.__delta ||
                value.octave != this.__octave;
            this.__chord = value.chord;
            this.__delta = value.delta;
            this.__octave = value.octave;
            this.__delta_dropdown.dropdown('set selected', this.__delta.toString());
            this.__octave_dropdown.dropdown('set selected', this.__octave.toString());
            if (should_update) this.update();
        } else
            this.__delta_dropdown.dropdown('set selected', '-1');
    }

    get payload() {
        return {
            note: this.note,
            enabled: this.enabled
        };
    }

    set payload(value) {
        this.note = value.note;
        this.enabled = value.enabled;
    }

    update() {
        // console.log('NoteWidget', 'update', this.chord, this.delta, this.octave, this.enabled, this.payload);
        const octave_back_color = chord_colors[this.__chord];
        this.__octave_dropdown.css('background-color', octave_back_color.css());

        const delta_back_color = delta_back_brighten(octave_back_color, this.__delta);
        const delta_front_color = delta_front_brighten(octave_back_color, this.__delta);
        this.__delta_dropdown.css('background-color', delta_back_color.css());
        this.__delta_dropdown.css('color', delta_front_color.css())

        this.__enabled_button.css('background-color', this.__enabled ? octave_back_color.css() : "black");

        if (this.onUpdate) this.onUpdate(this.payload);
    }

    ping(duration = 300) {
        this.elem.find('.icon.buttons')
            .css('margin-right', '10px')
            .animate({
                marginRight: 0
            }, duration);
    }
}
