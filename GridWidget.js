"use strict";

class GridWidget {
    constructor() {
        const menus = $($.parseHTML(`
        <div class="ui top attached menu">
            <div class="ui dropdown icon item">
                Chords
                <div class="menu">
                    <div class="set_chords_633 item">Pattern 633</div>
                    <div class="set_chords_444 item">Pattern 444</div>
                    <div class="divider"></div>
                    <div class="increment_chords item">Transpose +</div>
                    <div class="decrement_chords item">Transpose -</div>
                </div>
            </div>
            <div class="ui dropdown icon item">
                Octaves
                <div class="menu">
                    <div class="set_binary_octaves item">Repeat 0+</div>
                    <div class="clear_octaves item">All 0</div>
                </div>
            </div>
            <div class="ui dropdown icon item">
                Deltas
                <div class="menu">
                    <div class="set_all_first_deltas item">All 1st</div>
                    <div class="clear_deltas item">All &emptyset;</div>
                </div>
            </div>
            <div class="right menu">
                <div class="ui song search item">
                    <div class="ui transparent icon input">
                        <input class="prompt" type="text" placeholder="Search songs...">
                            <i class="search link icon search_all"></i>
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

        const chords_633 = [
            0, 0, 0, 0, 0, 0, 2, 2, 2, 4, 4, 4,
            0, 0, 0, 0, 0, 0, 2, 2, 2, 5, 5, 5,
            0, 0, 0, 0, 0, 0, 3, 3, 3, 5, 5, 5,
            1, 1, 1, 1, 1, 1, 3, 3, 3, 5, 5, 5
        ];

        const chords_444 = [];
        [0, 2, 4, 0, 2, 5, 0, 3, 5, 1, 3, 5].forEach((value, index) => {
            for (let kk = 0; kk < 4; kk++)
                chords_444.push(value);
        });

        this.widgets = chords_444.map((chord, index) => {
            const widget = new NoteWidget();
            widget.chord = chord;
            widget.onUpdate = () => {
                // console.log('widget.note', widget.note);
                // console.log('widget.index', widget.index);
                // console.log('widget.chord', widget.chord);
                this.score[index] = widget.note;
                this.update();
            }
            grid.append(widget.elem);
            return widget;
        });
        this.score = this.widgets.map(widget => widget.note);


        const widget_action = cb => () => this.widgets.forEach(cb);
        const set_action = (key, values) => widget_action((widget, index) => widget[key] = values[index]);

        const song_search = menus.find('.song.search');
        song_search.search({
            source: [{
                    title: 'NemaFull',
                    description: 'Nemamoussassa full song',
                    notes: nema_full(0, 2, 4),
                },
                {
                    title: 'NemaLeftHand',
                    description: 'Nemamoussassa left hand only',
                    notes: nema_left_hand(0, 2, 4),
                }
            ],
            minCharacters: 0,
            onSelect: (selection) => {
                // console.log('got song', selection.title, selection.notes)
                this.widgets.forEach((widget, index) => widget.note = selection.notes[index]);
            }
        });
        song_search.find('.search_all').click(() => {
            song_search.search('set value', '');
            song_search.search('query');
        })

        menus.find('.ui.dropdown').dropdown();

        { // chord tools
            const set_chords_action = chords => set_action('chord', chords);
            menus.find('.increment_chords').click(widget_action((widget, index) => widget.chord++));
            menus.find('.decrement_chords').click(widget_action((widget, index) => widget.chord += 6));
            menus.find('.set_chords_444').click(set_chords_action(chords_444));
            menus.find('.set_chords_633').click(set_chords_action(chords_633));
        }

        { // octave tools
            menus.find('.clear_octaves').click(widget_action((widget, index) => widget.octave = 5));
            menus.find('.set_binary_octaves').click(widget_action((widget, index) => widget.octave = index % 2 == 0 ? 5 : 6));
        }

        { // delta tools
            menus.find('.clear_deltas').click(widget_action((widget, index) => widget.delta = -1));
            menus.find('.set_all_first_deltas').click(widget_action((widget, index) => widget.delta = 0));
        }




        this.elem = $('<div>', {
            class: "ui segments"
        });
        this.elem.append(menus);
        this.elem.append(grid);
        this.elem.append(label);

        this.update();
    }

    update() {
        let sparse_score = this.score.map(elem => elem == null ? '__' : elem.note);
        sparse_score.splice(12, 0, "<br/>");
        sparse_score.splice(25, 0, "<br/>");
        sparse_score.splice(38, 0, "<br/>");
        sparse_score = sparse_score.join(' ');
        // console.log(sparse_score);

        this.elem.find('.score_label').html(sparse_score);
    }
}
