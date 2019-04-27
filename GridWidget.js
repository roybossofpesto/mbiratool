"use strict";

class GridWidget {
    constructor() {
        const menus = $($.parseHTML(`
        <div class="ui top attached menu">
            <div class="ui chords_menu dropdown icon item">
                Chords
                <div class="menu">
                    <div class="set_chords_one item">Pattern 633</div>
                    <div class="set_chords_two item">Pattern 444</div>
                    <div class="divider"></div>
                    <div class="increment_chords item">Transpose +</div>
                    <div class="decrement_chords item">Transpose -</div>
                </div>
            </div>
            <div class="ui octaves_menu dropdown icon item">
                Octaves
                <div class="menu">
                    <div class="set_binary_octaves item">Repeat 0+</div>
                    <div class="clear_octaves item">All 0</div>
                </div>
            </div>
            <div class="ui deltas_menu dropdown icon item">
                Deltas
                <div class="menu">
                    <div class="set_all_first_deltas item">All 1st</div>
                    <div class="clear_deltas item">All &emptyset;</div>
                </div>
            </div>
            <div class="ui deltas_menu dropdown icon item">
                Songs
                <div class="menu">
                    <div class="set_nemamoussassa item">Nemamoussassa</div>
                </div>
            </div>
            <!--<div class="ui edit_menu dropdown icon item">
                <i class="wrench icon"></i>
                <div class="menu">
                    <div class="item">
                        <i class=" dropdown icon"></i>
                        <span class="text">New</span>
                        <div class="menu">
                            <div class="item">Document</div>
                            <div class="item">Image</div>
                        </div>
                    </div>
                    <div class="item">
                        Open...
                    </div>
                    <div class="item">
                        Save...
                    </div>
                    <div class="item">Edit Permissions</div>
                    <div class="divider"></div>
                    <div class="header">
                        Export
                    </div>
                    <div class="item">
                        Share...
                    </div>
                </div>
            </div>-->
            <div class="right menu">
                <div class="ui right aligned category search item">
                    <div class="ui transparent icon input">
                        <input class="prompt" type="text" placeholder="Search animals...">
                        <i class="search link icon"></i>
                    </div>
                    <div class="results"></div>
                </div>
            </div>
        </div>
        `));
        const grid = $('<div>', {
            class: "ui twelve column center aligned grid segment",
        });
        const label = $('<div>', {
            class: "ui bottom attached left aligned score_label segment",
            style: "font-family: monospace;",
            text: "coucou",
        });

        const chords_one = [
            0, 0, 0, 0, 0, 0, 2, 2, 2, 4, 4, 4,
            0, 0, 0, 0, 0, 0, 2, 2, 2, 5, 5, 5,
            0, 0, 0, 0, 0, 0, 3, 3, 3, 5, 5, 5,
            1, 1, 1, 1, 1, 1, 3, 3, 3, 5, 5, 5
        ];

        const chords_two = [];
        [0, 2, 4, 0, 2, 5, 0, 3, 5, 1, 3, 5].forEach((value, index) => {
            for (let kk = 0; kk < 4; kk++)
                chords_two.push(value);
        });

        this.__widgets = chords_one.map((chord, index) => {
            const widget = new NoteWidget();
            widget.chord = chord;
            widget.onUpdate = () => {
                // console.log('widget.note', widget.note);
                // console.log('widget.index', widget.index);
                // console.log('widget.chord', widget.chord);
                this.__score[index] = widget.note;
                this.update();
            }
            grid.append(widget.elem);
            return widget;
        });
        this.__score = this.__widgets.map(widget => widget.note);

        menus.find('.ui.dropdown').dropdown();

        const widget_action = cb => () => this.__widgets.forEach(cb);

        // chord tools
        menus.find('.increment_chords').click(widget_action((widget, index) => widget.chord++));
        menus.find('.decrement_chords').click(widget_action((widget, index) => widget.chord--));
        menus.find('.set_chords_one').click(widget_action((widget, index) => widget.chord = chords_one[index]));
        menus.find('.set_chords_two').click(widget_action((widget, index) => widget.chord = chords_two[index]));

        // octave tools
        menus.find('.clear_octaves').click(widget_action((widget, index) => widget.octave = 5));
        menus.find('.set_binary_octaves').click(widget_action((widget, index) => widget.octave = index % 2 == 0 ? 5 : 6));

        // delta tools
        menus.find('.clear_deltas').click(widget_action((widget, index) => widget.delta = -1));
        menus.find('.set_all_first_deltas').click(widget_action((widget, index) => widget.delta = 0));

        // songs tools
        const nemamoussassa_notes = nema_full(0, 2, 5, 0);
        menus.find('.set_nemamoussassa').click(widget_action((widget, index) => widget.note = nemamoussassa_notes[index]));

        this.elem = $('<div>', {
            class: "ui segments"
        });
        this.elem.append(menus);
        this.elem.append(grid);
        this.elem.append(label);

        this.update();
    }

    update() {
        let sparse_score = this.__score.map(elem => elem == null ? '__' : elem.note);
        sparse_score.splice(12, 0, "<br/>");
        sparse_score.splice(25, 0, "<br/>");
        sparse_score.splice(38, 0, "<br/>");
        sparse_score = sparse_score.join(' ');
        // console.log(sparse_score);

        this.elem.find('.score_label').html(sparse_score);
    }
}
