/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015 Martin Eneqvist <marlun78@hotmail.com> (https://github.com/marlun78)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// Modification notice:
// This is a modified version of https://github.com/marlun78/number-to-words with the following improvements:
// - Matches the output of the `number_to_words` function in https://github.com/jaraco/inflect (used by the python library).
// - Removed all functions except `number_to_words(number)` to reduce the size of the code
// - Supports decimal numbers

const TEN = 10;
const ONE_HUNDRED = 100;
const ONE_THOUSAND = 1000;
const ONE_MILLION = 1000000;
const ONE_BILLION = 1000000000;
const ONE_TRILLION = 1000000000000;
const ONE_QUADRILLION = 1000000000000000;
const MAX = 9007199254740992;

const LESS_THAN_TWENTY = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];

const TENTHS_LESS_THAN_HUNDRED = ["zero", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

function isUnsafeNumber(num) {
    return num > Number.MAX_SAFE_INTEGER || num < Number.MIN_SAFE_INTEGER || num !== num; // checks for NaN
}

/**
 * Converts a number into words.
 * If the number is decimal, the decimals will be converted to words as well.
 * @example number_to_words(12.34) => 'twelve point three four'
 * @param {number} number
 * @returns {string}
 */
export function number_to_words(number) {
    if (isUnsafeNumber(number)) {
        throw new RangeError("Input is not a safe number, it's either too large or too small.");
    }
    if (Object.is(number, -0)) {
        // Handle the special case for -0.
        // NOTE: We cannot use === since `-0 === 0`.
        return "minus zero";
    }

    return number_to_words_helper(number);
}

function number_to_words_helper(number) {
    const numStr = number.toLocaleString("en-US", {
        useGrouping: false,
        maximumFractionDigits: 20,
    });

    // If negative, prepend “minus”
    if (numStr.startsWith("-")) {
        return "minus " + number_to_words_helper(numStr.slice(1));
    }
    const [integer, fractional, ...rest] = numStr.split(".");
    if (rest.length > 0) {
        throw new TypeError("Invalid number: " + number + " (" + typeof number + ")");
    }

    const integerPart = parseInt(integer, 10);
    const words = to_words(integerPart);
    if (fractional !== undefined) {
        let decimals = fractional.replace(/0+$/, ""); // Remove trailing zeros
        if (decimals.length > 0) {
            decimals = decimals
                .split("")
                .map((digit) => LESS_THAN_TWENTY[parseInt(digit, 10)])
                .join(" ");
        } else {
            decimals = "zero";
        }
        return words + " point " + decimals;
    }

    return words;
}
function to_words(number, words) {
    let remainder, word;

    // We're done
    if (number === 0) {
        return !words ? "zero" : words.join(" ");
    }
    words ??= [];

    if (number < 20) {
        remainder = 0;
        word = LESS_THAN_TWENTY[number];
    } else if (number < ONE_HUNDRED) {
        remainder = number % TEN;
        word = TENTHS_LESS_THAN_HUNDRED[Math.floor(number / TEN)];
        // Handle remainder for tens
        if (remainder) {
            word += "-" + LESS_THAN_TWENTY[remainder];
            remainder = 0;
        }
    } else if (number < ONE_THOUSAND) {
        remainder = number % ONE_HUNDRED;
        word = to_words(Math.floor(number / ONE_HUNDRED)) + " hundred";
    } else if (number < ONE_MILLION) {
        remainder = number % ONE_THOUSAND;
        word = to_words(Math.floor(number / ONE_THOUSAND)) + " thousand";
    } else if (number < ONE_BILLION) {
        remainder = number % ONE_MILLION;
        word = to_words(Math.floor(number / ONE_MILLION)) + " million";
    } else if (number < ONE_TRILLION) {
        remainder = number % ONE_BILLION;
        word = to_words(Math.floor(number / ONE_BILLION)) + " billion";
    } else if (number < ONE_QUADRILLION) {
        remainder = number % ONE_TRILLION;
        word = to_words(Math.floor(number / ONE_TRILLION)) + " trillion";
    } else if (number <= MAX) {
        remainder = number % ONE_QUADRILLION;
        word = to_words(Math.floor(number / ONE_QUADRILLION)) + " quadrillion";
    }

    if (remainder && remainder < ONE_HUNDRED) {
        word += " and " + to_words(remainder);
        remainder = 0;
    } else if (remainder) {
        word += ", " + to_words(remainder);
        remainder = 0;
    }

    words.push(word);
    return to_words(remainder, words);
}
