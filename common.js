'use strict';

const wrap = xx => xx % 7;
const letters = "CDEFGAB";

// create note from chord, delta (=0 root, =2 third, =4 fifth) and octave on keyboard
// delta < 0 return an empty note (null)
const create_note = (chord, delta, octave) => delta < 0 ? null : {
    note: `${letters[wrap(chord+delta)]}${octave}`,
    chord: wrap(chord),
    octave: octave,
    delta: delta,
};

const chord_colors = ["#1c96fe", "#feb831", "#aee742", "#b75ac4", "#15cdc2", "#fa2424", "#ff5986"]
    .map(color => chroma(color));

const delta_brighten = (color, delta) => color.brighten(
    delta == 4 ? 1 :
    delta == 3 ? 1.7 :
    delta == 2 ? 2.5 :
    delta == 0 ? 0 :
    -10);

// adapted from lodash
const debounce = (func, wait, options) => {
    let lastArgs,
        lastThis,
        maxWait,
        result,
        timerId,
        lastCallTime,
        lastInvokeTime = 0,
        leading = false,
        maxing = false,
        trailing = true;
    if (typeof func !== 'function') {
        throw new TypeError(FUNC_ERROR_TEXT);
    }
    wait = Number(wait) || 0;
    if (typeof options === 'object') {
        leading = !!options.leading;
        maxing = 'maxWait' in options;
        maxWait = maxing ?
            Math.max(Number(options.maxWait) || 0, wait) :
            maxWait;
        trailing = 'trailing' in options ?
            !!options.trailing :
            trailing;
    }

    function invokeFunc(time) {
        let args = lastArgs,
            thisArg = lastThis;

        lastArgs = lastThis = undefined;
        lastInvokeTime = time;
        result = func.apply(thisArg, args);
        return result;
    }

    function leadingEdge(time) {
        // Reset any `maxWait` timer.
        lastInvokeTime = time;
        // Start the timer for the trailing edge.
        timerId = setTimeout(timerExpired, wait);
        // Invoke the leading edge.
        return leading ?
            invokeFunc(time) :
            result;
    }

    function remainingWait(time) {
        let timeSinceLastCall = time - lastCallTime,
            timeSinceLastInvoke = time - lastInvokeTime,
            result = wait - timeSinceLastCall;
        return maxing ?
            Math.min(result, maxWait - timeSinceLastInvoke) :
            result;
    }

    function shouldInvoke(time) {
        let timeSinceLastCall = time - lastCallTime,
            timeSinceLastInvoke = time - lastInvokeTime;
        // Either this is the first call, activity has stopped and we're at the trailing
        // edge, the system time has gone backwards and we're treating it as the
        // trailing edge, or we've hit the `maxWait` limit.
        return (lastCallTime === undefined || (timeSinceLastCall >= wait) || (timeSinceLastCall < 0) || (maxing && timeSinceLastInvoke >= maxWait));
    }

    function timerExpired() {
        const time = Date.now();
        if (shouldInvoke(time)) {
            return trailingEdge(time);
        }
        // Restart the timer.
        timerId = setTimeout(timerExpired, remainingWait(time));
    }

    function trailingEdge(time) {
        timerId = undefined;

        // Only invoke if we have `lastArgs` which means `func` has been debounced at
        // least once.
        if (trailing && lastArgs) {
            return invokeFunc(time);
        }
        lastArgs = lastThis = undefined;
        return result;
    }

    function cancel() {
        if (timerId !== undefined) {
            clearTimeout(timerId);
        }
        lastInvokeTime = 0;
        lastArgs = lastCallTime = lastThis = timerId = undefined;
    }

    function flush() {
        return timerId === undefined ?
            result :
            trailingEdge(Date.now());
    }

    function debounced() {
        let time = Date.now(),
            isInvoking = shouldInvoke(time);
        lastArgs = arguments;
        lastThis = this;
        lastCallTime = time;

        if (isInvoking) {
            if (timerId === undefined) {
                return leadingEdge(lastCallTime);
            }
            if (maxing) {
                // Handle invocations in a tight loop.
                timerId = setTimeout(timerExpired, wait);
                return invokeFunc(lastCallTime);
            }
        }
        if (timerId === undefined) {
            timerId = setTimeout(timerExpired, wait);
        }
        return result;
    }
    debounced.cancel = cancel;
    debounced.flush = flush;
    return debounced;
}
