import { describe, expect, test } from "vitest";
import { number_to_words } from "../outetts.js/version/v1/utils/number_to_words.js";

const TEST_CASES = {
    "Basic integers": [
        [0, "zero"],
        [1, "one"],
        [5, "five"],
        [42, "forty-two"],
    ],
    "Large integers": [
        [1000, "one thousand"],
        [1234567890, "one billion, two hundred and thirty-four million, five hundred and sixty-seven thousand, eight hundred and ninety"],
        [9876543210, "nine billion, eight hundred and seventy-six million, five hundred and forty-three thousand, two hundred and ten"],
    ],
    "Triple-digit numbers": [
        [100, "one hundred"],
        [123, "one hundred and twenty-three"],
        [999, "nine hundred and ninety-nine"],
    ],
    Decimals: [
        [1, "one"],
        [1.0, "one"],
        [0.5, "zero point five"],
        [123.45, "one hundred and twenty-three point four five"],
        [3.1415926535, "three point one four one five nine two six five three five"],
        [0.0001, "zero point zero zero zero one"],
        [0.1000000023, "zero point one zero zero zero zero zero zero zero two three"],
        [0.987654, "zero point nine eight seven six five four"],
        [9.87654, "nine point eight seven six five four"],
        [98.7654, "ninety-eight point seven six five four"],
        [987.654, "nine hundred and eighty-seven point six five four"],
        [9876.54, "nine thousand, eight hundred and seventy-six point five four"],
        [98765.4, "ninety-eight thousand, seven hundred and sixty-five point four"],
    ],
    "Negative numbers": [
        [-1, "minus one"],
        [-123, "minus one hundred and twenty-three"],
        [-0.5, "minus zero point five"],
        [-999, "minus nine hundred and ninety-nine"],
    ],
    "Edge cases": [
        [0, "zero"],
        [-0, "minus zero"],
        [999999999999, "nine hundred and ninety-nine billion, nine hundred and ninety-nine million, nine hundred and ninety-nine thousand, nine hundred and ninety-nine"],
        [1000000000000, "one trillion"],
    ],
    "Single digits": [
        [0, "zero"],
        [1, "one"],
        [2, "two"],
        [3, "three"],
        [4, "four"],
        [5, "five"],
        [6, "six"],
        [7, "seven"],
        [8, "eight"],
        [9, "nine"],
    ],
    "Two-digit numbers": [
        [11, "eleven"],
        [20, "twenty"],
        [99, "ninety-nine"],
        [81, "eighty-one"],
    ],
    "Special cases for teens": [
        [13, "thirteen"],
        [19, "nineteen"],
        [15, "fifteen"],
        [18, "eighteen"],
    ],
    "Round numbers": [
        [10000, "ten thousand"],
        [1000000, "one million"],
        [1000000000, "one billion"],
        [1000000000000, "one trillion"],
    ],
    "Mixed digits": [
        [105, "one hundred and five"],
        [1001, "one thousand and one"],
        [1000001, "one million and one"],
        [1000000001, "one billion and one"],
        [1001000000001, "one trillion, one billion and one"],
        [1001000000100, "one trillion, one billion, one hundred"],
        [101010, "one hundred and one thousand and ten"],
        [555123, "five hundred and fifty-five thousand, one hundred and twenty-three"],
    ],
    "Very small numbers": [
        [0.0001, "zero point zero zero zero one"],
        [0.0000001, "zero point zero zero zero zero zero zero one"],
    ],
};

describe("Number to Words Tests", () => {
    Object.keys(TEST_CASES).forEach((testName) => {
        test(testName, () => {
            TEST_CASES[testName].forEach(([input, expected]) => {
                expect(number_to_words(input)).toEqual(expected);
            });
        });
    });
});

// Expect errors for unsafe numbers
test("Unsafe numbers", () => {
    expect(() => number_to_words(Number.MAX_SAFE_INTEGER + 1)).toThrow();
    expect(() => number_to_words(Number.MIN_SAFE_INTEGER - 1)).toThrow();
    expect(() => number_to_words(Number.MAX_VALUE)).toThrow();
    expect(() => number_to_words(NaN)).toThrow();
    expect(() => number_to_words(Infinity)).toThrow();
    expect(() => number_to_words(-Infinity)).toThrow();
});
