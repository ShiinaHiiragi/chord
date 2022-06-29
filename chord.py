import re
import math

from fuzzy_expert.variable import FuzzyVariable
from fuzzy_expert.rule import FuzzyRule
from fuzzy_expert.inference import DecompositionalInference


def match_list(left: list, right: list) -> bool:
    assert type(left) == list
    assert type(right) == list
    if len(left) != len(right):
        return False

    size: int = len(left)
    for index in range(0, size):
        if left[index] != right[index]:
            return False
    return True

# example:
#   database = [[1, 2], [2, 3], [3, 4]]
#   sample = [2, 3]
#   then return [2, 3]
def find_list(database: list[list[any]], sample: list[any]) -> bool:
    for item in database:
        if match_list(item, sample):
            return True
    return False


# this class cannot be deleted
class Portion:
    pass


class Portion:
    def __init__(self, count: int, total: int):
        self._count = count
        self._total = total

        assert type(count) == int
        assert type(total) == int
        assert count <= total

    def add(self, num: int = 1) -> None:
        assert self._count + num <= self._total
        self._count += num

    def accumulate(self, delta: Portion) -> None:
        self._count += delta.get_count()
        self._total += delta.get_total()

    def __float__(self) -> float:
        return float(self._count / self._total)

    def __str__(self) -> str:
        return "[%s, %s]" % (self._count, self._total)

    def __repr__(self) -> str:
        return self.__str__()

    def get_count(self) -> int:
        return self._count

    def get_total(self) -> int:
        return self._total


class Note:
    KEY_MAP: dict = {
        "C": -24,
        "D": -22,
        "E": -20,
        "F": -19,
        "G": -17,
        "A": -15,
        "B": -13,
        "c": -12,
        "d": -10,
        "e": -8,
        "f": -7,
        "g": -5,
        "a": -3,
        "b": -1
    }

    def __init__(self, sign: str):
        self._note: int = 0
        base: int = 1
        assert type(sign) == str

        pattern: re.Match = re.match("^([♯♭]?)([A-Ga-g])('*)$", sign)
        assert pattern != None

        # the input should match the rule
        offset: str = pattern.group(1)
        clef: str = pattern.group(2)
        group: str = pattern.group(3)

        # calculate the offset relative to center c
        self._note = Note.KEY_MAP[clef]
        if re.match("[A-G]", clef):
            base = -1
        self._note += len(group) * base * 12
        if offset == "♯":
            self._note += 1
        elif offset == "♭":
            self._note -= 1

        # the input should be in the range of piano
        assert self._note >= -39 and self._note <= 48

    @staticmethod
    def round_delta(note: int) -> int:
        while (note >= 12):
            note -= 12
        while (note <= -12):
            note += 12
        return note

    @staticmethod
    def diff(model: list[int], sample: list[int]) -> int:
        result: int = 0
        size: int = len(sample)
        for index in range(0, size):
            pair: list[int] = [model[index]]
            if pair[0] > 0:
                pair.append(pair[0] - 12)
            elif pair[0] < 0:
                pair.append(pair[0] + 12)
            else:
                pair.append(0)
            result += min((pair[0] - sample[index])**2, (pair[1] - sample[index])**2)

        return result

    @staticmethod
    def interval(left: int, right: int, mod: bool = False) -> int:
        result = abs(left - right)
        return result % 12 if mod else result

    @staticmethod
    def is_octave(left: int, right: int) -> bool:
        return Note.interval(left, right, mod = True) == 0

    @staticmethod
    def trans(base: int, note: int) -> bool:
        while note < base:
            note += 12
        while note >= base + 12:
            note -= 12
        return note

    def __str__(self) -> str:
        return str(self.get_note())

    def __repr__(self) -> str:
        return self.__str__()

    def get_note(self) -> int:
        return self._note


class Scale():
    KEY_MAP: dict = { "C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11 }
    NATURAL_MAJOR: list[int] = [2, 2, 1, 2, 2, 2, 1]
    NATURAL_MINOR: list[int] = [2, 1, 2, 2, 1, 2, 2]
    MELODIC_MINOR: list[int] = [2, 1, 2, 2, 2, 2, 1]
    HARMONIC_MINOR: list[int] = [2, 1, 2, 2, 1, 3, 1]

    def __init__(self, tonal: str):
        self._tonal: str = tonal
        self._scale: list[int] = [0] * 14
        self._traid: list[list[int]] = [[0] * 3 for index in range(0, 7)]
        self._seven: list[list[int]] = [[0] * 4 for index in range(0, 7)]
        assert type(tonal) == str

        # C: C natural major
        # c: c natural minor
        # cm: c melodic minor
        # ch: c harmonic minor
        pattern: re.Match = re.match("^([♯♭]?)([A-G]|[a-g][hm]?)$", tonal)
        assert pattern != None

        # get the tonal of scale
        offset: str = pattern.group(1)
        clef: str = pattern.group(2)
        self._scale[0] = Scale.KEY_MAP[clef[0].upper()]
        if offset == "♯":
            self._scale[0] += 1
        elif offset == "♭":
            self._scale[0] -= 1

        # complete scale
        base: list[int] = None
        if re.match("[A-G]", clef):
            base = Scale.NATURAL_MAJOR
        elif re.match("[a-g]h", clef):
            base = Scale.HARMONIC_MINOR
        elif re.match("[a-g]m", clef):
            base = Scale.MELODIC_MINOR
        else:
            base = Scale.NATURAL_MINOR

        for index in range(1, 8):
            self._scale[index] = self._scale[index - 1] + base[index - 1]
        for index in range(8, 14):
            self._scale[index] = self._scale[index - 1] + base[index - 8]
        for index in range(0, 7):
            for sub_index in range(0, 3):
                self._traid[index][sub_index] = self._scale[index + 2 * sub_index]
                self._seven[index][sub_index] = self._scale[index + 2 * sub_index]
            self._seven[index][3] = self._scale[index + 6]

    def _fit_chord(self, typing: int, chord: list[int]) -> bool:
        if typing == 3:
            return find_list(self._traid, chord)
        elif typing == 4:
            return find_list(self._seven, chord)
        else:
            return False

    def __str__(self) -> str:
        return self.get_tonal()

    def __repr__(self) -> str:
        return self.__str__()

    def get_tonic(self) -> int:
        return self._scale[0]

    def get_dominant(self) -> int:
        return self._scale[4]

    def get_tonal(self) -> str:
        return self._tonal

    def get_scale(self) -> list[int]:
        return self._scale

    def get_traid(self) -> list[list[int]]:
        return self._traid

    def get_seven(self) -> list[list[int]]:
        return self._seven


class Chord():
    PERFECT_CONSONANCES: list[int] = [0, 5, 7, 12]
    IMPERFECT_CONSONANCES: list[int] = [3, 4, 8, 9]
    PERFECT_DISSONANCES: list[int] = [1, 2, 6, 10, 11]
    TRAID_CONSONANCES: list[list[int]] = [
        [4, 3],       # major traid
        [3, 4]        # minor traid
    ]

    TRAID_DISSONANCES: list[list[int]] = [
        [4, 4],       # augmented triad
        [3, 3]        # diminished traid
    ]

    SEVEN_CONSONANCES: list[list[int]] = [
        [3, 3, 4],    # half diminished seventh chord
        [3, 4, 3],    # minor seventh chord
        [4, 3, 3],    # dominant seventh chord
        [4, 3, 4]     # major seventh chord
    ]

    SEVEN_SPECIAL: list[list[int]] = [
        [3, 3, 3]     # diminished seventh chord
    ]

    SEVEN_DISSONANCES: list[list[int]] = [
        [3, 4, 4],    # minor major seventh chord
        [4, 4, 2],    # augmented seventh chord
        [4, 4, 3]     # augmented major seventh chord
    ]

    def __init__(self, tonal: str, chord: list[str]):
        self._scale: Scale = Scale(tonal)
        self._chord: list[int] = [None] * len(chord)
        assert type(tonal) == str
        assert type(chord) == list

        for index, item in enumerate(chord):
            assert type(item) == str
            self._chord[index] = Note(item).get_note()

    def perfect_consonances(self) -> Portion:
        return self._find_pair(Chord.PERFECT_CONSONANCES)

    def imperfect_consonances(self) -> Portion:
        return self._find_pair(Chord.IMPERFECT_CONSONANCES)

    def perfect_dissonances(self) -> Portion:
        return self._find_pair(Chord.PERFECT_DISSONANCES)

    def _find_pair(self, target: list[int]) -> Portion:
        counter: int = 0
        size: int = len(self._chord)
        for index in range(0, size):
            for sub_index in range(index + 1, size):
                if Note.interval(self._chord[index], self._chord[sub_index], mod = True) in target:
                    counter += 1
        return Portion(counter, int(size * (size - 1) / 2))

    def _fit_type(self) -> int:
        result: list[int] = []
        for item in self._chord:
            handle = lambda result_item: Note.is_octave(item, result_item)
            if len(list(filter(handle, result))) == 0:
                result.append(item)
        return len(result)

    def _fit_interval(self, typing: int, chord: list[int]) -> int:
        deltas: list[int] = []
        for index in range(1, typing):
            deltas.append(chord[index] - chord[index - 1])

        if typing == 3:
            if find_list(Chord.TRAID_CONSONANCES, deltas):
                return id(Chord.TRAID_CONSONANCES)
            elif find_list(Chord.TRAID_DISSONANCES, deltas):
                return id(Chord.TRAID_DISSONANCES)
            else:
                return None
        elif typing == 4:
            if find_list(Chord.SEVEN_CONSONANCES, deltas):
                return id(Chord.SEVEN_CONSONANCES)
            elif find_list(Chord.SEVEN_DISSONANCES, deltas):
                return id(Chord.SEVEN_DISSONANCES)
            elif find_list(Chord.SEVEN_SPECIAL, deltas):
                return id(Chord.SEVEN_SPECIAL)
            else:
                return None
        else:
            return None

    def _fit_scale(self, typing: int) -> list[any]:
        if not typing in [3, 4]:
            return None

        for item in self._chord:
            chord: list[int] = []
            for sub_item in self._scale.get_scale()[:7]:
                if Note.is_octave(item, sub_item):
                    chord.append(sub_item)

            if len(chord) == 0:
                return None

            for sub_item in self._chord:
                chord.append(Note.trans(chord[0], sub_item))
            chord = list(set(chord))
            chord.sort()
            if self._scale._fit_chord(typing, chord) != False:
                return [chord, self._fit_interval(typing, chord)]
        return None

    def _fit_chord(self, typing: int) -> list[any]:
        for item in self._chord:
            chord: list[int] = [Note.trans(self._scale.get_tonic(), item)]
            for sub_item in self._chord:
                chord.append(Note.trans(chord[0], sub_item))
            chord = list(set(chord))
            chord.sort()
            interval: int = self._fit_interval(typing, chord)
            if interval != None:
                return [chord, interval]
        return None

    def __str__(self) -> str:
        return str(self.get_chord())

    def __repr__(self) -> str:
        return self.__str__()

    def get_tonal(self) -> str:
        return self._scale.get_tonal()

    def get_chord(self) -> list[int]:
        return self._chord

    def get_bass(self) -> int:
        return min(self._chord)


class Harmony:
    def __init__(self, tonal: str, chords: list[list[str]]):
        assert type(chords) == list
        self._size = len(chords)
        self._scale: Scale = Scale(tonal)
        self._chords: list[Chord] = [None] * self._size
        for index, item in enumerate(chords):
            self._chords[index] = Chord(tonal, item)
        self._find_roots()

    def perfect_consonances(self) -> Portion:
        result: Portion = Portion(0, 0)
        for item in self._chords:
            result.accumulate(item.perfect_consonances())
        return result

    def imperfect_consonances(self) -> Portion:
        result: Portion = Portion(0, 0)
        for item in self._chords:
            result.accumulate(item.imperfect_consonances())
        return result

    def perfect_dissonances(self) -> Portion:
        result: Portion = Portion(0, 0)
        for item in self._chords:
            result.accumulate(item.perfect_dissonances())
        return result

    def authentic_cadence(self) -> bool:
        return Note.is_octave(self._roots[-1], self._scale.get_tonic())

    def half_cadence(self) -> bool:
        return Note.is_octave(self._roots[-1], self._scale.get_dominant())

    def perfect_cadence(self) -> bool:
        return Note.is_octave(self._roots[-1], self._chords[-1].get_bass())

    def perfect_cadence(self) -> bool:
        return Note.is_octave(self._roots[-1], self._chords[-1].get_bass())

    def fifth(self) -> Portion:
        result: Portion = Portion(0, self._size - 1)
        for index in range(1, self._size):
            if Note.interval(self._roots[index - 1], self._roots[index], mod = True) in [5, 7]:
                result.add()
        return result

    def chord_portion(self, chord: list[int]) -> Portion:
        result: Portion = Portion(0, self._size)
        assert type(chord) == list

        for item in self._dense:
            deltas: list[int] = []
            for index in range(1, len(item)):
                deltas.append(item[index] - item[index - 1])
            if match_list(deltas, chord):
                result.add()
        return result

    def progression_difference(self, chord: list[str]) -> int:
        result: int = 121 * (self._size - 1)
        assert type(chord) == list

        size: int = len(chord)
        progression: list[int] = [Note(item).get_note() for item in chord]
        progression_deltas: list[int] = []
        roots_deltas: list[int] = []
        for index in range(1, size):
            progression_deltas.append(progression[index] - progression[index - 1])
        for index in range(1, self._size):
            roots_deltas.append(Note.round_delta(self._roots[index] - self._roots[index - 1]))

        # increase the length of deltas
        progression_deltas = progression_deltas * math.ceil((size + self._size - 2) / (size - 1))
        for index in range(0, size - 1):
            result = min(result, Note.diff(progression_deltas[index: index + self._size - 1], roots_deltas))
        return result

    def _find_roots(self) -> None:
        self._roots: list[int] = [None] * self._size
        self._dense: list[list[int]] = [None] * self._size
        self._consonances: Portion = Portion(0, self._size)
        self._dissonances: Portion = Portion(0, self._size)
        self._errors: Portion = Portion(0, self._size)

        for index in range(0, self._size):
            current_chord: Chord = self._chords[index]
            typing: int = current_chord._fit_type()
            fit_result: list[any] = current_chord._fit_scale(typing)
            if fit_result == None:
                fit_result = current_chord._fit_chord(typing)

            if fit_result != None:
                self._dense[index] = fit_result[0]
                if fit_result[1] in [id(Chord.TRAID_CONSONANCES), id(Chord.SEVEN_CONSONANCES)]:
                    self._roots[index] = Note.trans(current_chord.get_bass(), fit_result[0][0])
                    self._consonances.add()
                elif fit_result[1] in [id(Chord.TRAID_DISSONANCES), id(Chord.SEVEN_DISSONANCES)]:
                    self._roots[index] = Note.trans(current_chord.get_bass(), fit_result[0][0])
                    self._dissonances.add()
                elif fit_result[1] == id(Chord.SEVEN_SPECIAL):
                    # the root of dominant seventh chord cannot be clarified by analyzer
                    self._roots[index] = current_chord.get_bass()
                    self._consonances.add()
                next
            else:
                self._roots[index] = current_chord.get_bass()
                self._errors.add()

    def __str__(self) -> str:
        return str(self.get_chords())

    def __repr__(self) -> str:
        return self.__str__()

    def get_size(self) -> int:
        return self._size

    def get_tonal(self) -> str:
        return self._scale.get_tonal()

    def get_chords(self) -> list[Chord]:
        return self._chords

    def get_roots(self) -> list[int]:
        return self._roots

    def get_dense(self) -> list[list[int]]:
        return self._dense

    def get_consonances(self) -> Portion:
        return self._consonances

    def get_dissonances(self) -> Portion:
        return self._dissonances

    def get_errors(self) -> Portion:
        return self._errors

class Client:
    RATIO: FuzzyVariable = FuzzyVariable(
        universe_range=(0.0, 1.0),
        terms={
            "HIGH": [(0.0, 0.0), (0.3, 0.0), (0.7, 1.0), (1.0, 1.0)],
            "MEDIUM": [(0.0, 0.0), (0.1, 0.0), (0.5, 1.0), (1.0, 1.0)],
            "LOW": [(0.0, 1.0), (0.3, 1.0), (0.7, 0.0), (1.0, 0.0)]
        }
    )

    BOOL: FuzzyVariable = FuzzyVariable(
        universe_range=(0.0, 1.0),
        terms={
            "TRUE": [(0.0, 0.0), (0.49, 0.0), (0.51, 1.0), (1.0, 1.0)],
            "FALSE": [(0.0, 1.0), (0.49, 1.0), (0.51, 0.0), (1.0, 0.0)]
        }
    )

    DIFF: FuzzyVariable = FuzzyVariable(
        universe_range=(0.0, 11.0),
        terms={
            "UNLIKE": ('smf', 0.0, 3.0),
            "ALIKE": ('zmf', 0.0, 3.0)
        }
    )

    DEGREE: FuzzyVariable = FuzzyVariable(
        universe_range=(0.0, 1.0),
        terms={
            "GREAT": [(0.0, 0.0), (1.0, 1.0)],
            "WELL": [(0.0, 0.0), (0.5, 1.0), (1.0, 1.0)],
            "LITTLE": [(0.0, 1.0), (1.0, 0.0)]
        }
    )

    VARIABLES: dict[str, FuzzyVariable] = {
        "CONSONANCES_PAIR_PERCENTAGE": RATIO,
        "CONSONANCES_CHORD_PERCENTAGE": RATIO,
        "DIMINISHED_SEVENTH_CHORD_PERCENTAGE": RATIO,
        "NORMATIVE_CADENCE": BOOL,
        "PERFECT_CADENCE": BOOL,
        "FIFTH_PROGRESSION_PERCENTAGE": RATIO,
        "CANON_HARMONY_DIFFERENCES": DIFF,
        "POP_HARMONY_DIFFERENCES": DIFF,
        "DEGREE_OF_NORMATIVE": DEGREE,
        "DEGREE_OF_HARMONY": DEGREE,
        "DEGREE_OF_COMMONNESS": DEGREE,
        "DEGREE_OF_TENSION": DEGREE,
    }

    RULES = [
        FuzzyRule(
            premise=[
                ("CONSONANCES_PAIR_PERCENTAGE", "HIGH"),
                ("AND", "CONSONANCES_CHORD_PERCENTAGE", "HIGH"),
                ("AND", "NORMATIVE_CADENCE", "TRUE"),
            ],
            consequence=[("DEGREE_OF_NORMATIVE", "GREAT")]
        ),
        FuzzyRule(
            premise=[
                ("CONSONANCES_PAIR_PERCENTAGE", "LOW"),
                ("AND", "CONSONANCES_CHORD_PERCENTAGE", "LOW"),
                ("AND", "NORMATIVE_CADENCE", "FALSE"),
            ],
            consequence=[("DEGREE_OF_NORMATIVE", "LITTLE")]
        ),
        FuzzyRule(
            premise=[
                ("CONSONANCES_PAIR_PERCENTAGE", "HIGH"),
                ("AND", "CONSONANCES_CHORD_PERCENTAGE", "HIGH"),
                ("AND", "NORMATIVE_CADENCE", "TRUE"),
                ("AND", "PERFECT_CADENCE", "TRUE"),
                ("AND", "FIFTH_PROGRESSION_PERCENTAGE", "MEDIUM"),
            ],
            consequence=[("DEGREE_OF_HARMONY", "GREAT")]
        ),
        FuzzyRule(
            premise=[
                ("CONSONANCES_PAIR_PERCENTAGE", "LOW"),
                ("AND", "CONSONANCES_CHORD_PERCENTAGE", "LOW"),
                ("AND", "NORMATIVE_CADENCE", "FALSE"),
                ("AND", "PERFECT_CADENCE", "FALSE"),
                ("AND", "FIFTH_PROGRESSION_PERCENTAGE", "LOW"),
            ],
            consequence=[("DEGREE_OF_HARMONY", "LITTLE")]
        ),
        FuzzyRule(
            premise=[
                ("CANON_HARMONY_DIFFERENCES", "ALIKE"),
                ("OR", "POP_HARMONY_DIFFERENCES", "ALIKE"),
            ],
            consequence=[("DEGREE_OF_COMMONNESS", "GREAT")]
        ),
        FuzzyRule(
            premise=[
                ("CANON_HARMONY_DIFFERENCES", "UNLIKE"),
                ("OR", "POP_HARMONY_DIFFERENCES", "UNLIKE"),
            ],
            consequence=[("DEGREE_OF_COMMONNESS", "LITTLE")]
        ),
        FuzzyRule(
            premise=[
                ("DIMINISHED_SEVENTH_CHORD_PERCENTAGE", "MEDIUM")
            ],
            consequence=[("DEGREE_OF_TENSION", "WELL")]
        ),
        FuzzyRule(
            premise=[
                ("DIMINISHED_SEVENTH_CHORD_PERCENTAGE", "LOW")
            ],
            consequence=[("DEGREE_OF_TENSION", "LITTLE")]
        ),
    ]

    def __init__(self, harmony: Harmony):
        self._harmony: Harmony = harmony
        self._model: tuple = DecompositionalInference(
            and_operator = "prod",
            or_operator = "max",
            implication_operator = "Rc",
            composition_operator = "max-prod",
            production_link = "max",
            defuzzification_operator = "cog",
        )(
            variables = Client.VARIABLES,
            rules = Client.RULES,
            CONSONANCES_PAIR_PERCENTAGE = self.consonances_pair_percentage(),
            CONSONANCES_CHORD_PERCENTAGE = self.consonances_chord_percentage(),
            DIMINISHED_SEVENTH_CHORD_PERCENTAGE = self.specific_chord_percentage([3, 3, 3]),
            NORMATIVE_CADENCE = self.normative_cadence(),
            PERFECT_CADENCE = self.perfect_cadence(),
            FIFTH_PROGRESSION_PERCENTAGE = self.fifth_progression_percentage(),
            CANON_HARMONY_DIFFERENCES = self.specific_harmony_difference(["c''", "b'", "a'", "g'", "f'", "e'", "d'", "g'"]),
            POP_HARMONY_DIFFERENCES = self.specific_harmony_difference(["f'", "g'", "e'", "a'", "d'", "g'", "c'"]),
        )

    @staticmethod
    def to_float(input: any) -> float:
        return float('%.1f' % float(input))

    def consonances_pair_percentage(self) -> float:
        return Client.to_float(
            float(self._harmony.perfect_consonances()) +
            float(self._harmony.imperfect_consonances())
        )

    def consonances_chord_percentage(self) -> float:
        return Client.to_float(self._harmony.get_consonances())

    def specific_chord_percentage(self, chord: list[int]) -> float:
        return Client.to_float(self._harmony.chord_portion(chord))

    def normative_cadence(self) -> float:
        return Client.to_float(self._harmony.authentic_cadence() or self._harmony.half_cadence())

    def perfect_cadence(self) -> float:
        return Client.to_float(self._harmony.perfect_cadence())

    def fifth_progression_percentage(self) -> float:
        return Client.to_float(self._harmony.fifth())

    def specific_harmony_difference(self, harmony: list[str]) -> float:
        return round(
            math.sqrt(
                self._harmony.progression_difference(harmony) /
                (self._harmony.get_size() - 1)
            )
        )

    def get_model_result(self) -> tuple:
        return self._model


if __name__ == "__main__":
    chorale: Harmony = Harmony(
        tonal = "C",
        chords = [
            ["c", "g", "e'", "c''"],
            ["c", "g", "e'", "c''"],
            ["e", "g", "c'", "c''"],
            ["g", "b", "d'", "g'"],
            ["e", "e'", "g'", "b'"],
            ["c", "e'", "g'", "c''"],
            ["d", "d'", "♯f'", "a'"],
            ["G", "b", "d'", "g'"]
        ]
    )

    chorale_client: Client = Client(chorale)
    print(chorale_client.get_model_result())

    sonata: Harmony = Harmony(
        tonal = "c",
        chords = [
            ["c'", "♯f'", "a'", "♭e''"],
            ["♭b", "g'", "d''"],
            ["c'", "♯f'", "a'", "♭e''"],
            ["b", "♯f'", "a'", "♯d''"],
            ["e'", "g'", "b'", "e''"]
        ]
    )

    sonata_client: Client = Client(sonata)
    print(sonata_client.get_model_result())
