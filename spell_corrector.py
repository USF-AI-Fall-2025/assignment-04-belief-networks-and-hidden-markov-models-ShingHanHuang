import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

END_STATE = "<END>"


def load_training_pairs(path: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            correct, typed_part = line.split(":", 1)
            correct = correct.strip()
            typed_variants = typed_part.strip().split()
            for typed in typed_variants:
                typed = typed.strip()
                if typed:
                    pairs.append((correct, typed))
    return pairs
class SpellCheckerHMM:
    def __init__(self) -> None:
        self.states: List[str] = []
        self.state_set: Set[str] = set()

        self.start_counts: Counter[str] = Counter()
        self.start_total = 0

        self.transition_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        self.transition_totals: Counter[str] = Counter()

        self.emission_counts: Dict[str, Counter[str]] = defaultdict(Counter)
        self.emission_totals: Counter[str] = Counter()

        self.start_log_probs: Dict[str, float] = {}
        self.transition_log_probs: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.emission_log_probs: Dict[str, Dict[str, float]] = defaultdict(dict)

    def train(self, pairs: Iterable[Tuple[str, str]]) -> None:
        for correct_raw, typed_raw in pairs:
            correct = correct_raw.strip().lower()
            typed = typed_raw.strip().lower()
            if not correct or not typed:
                continue

            self.state_set.update(correct)

            letters = list(correct)
            if not letters:
                continue

            self.start_counts[letters[0]] += 1
            self.start_total += 1

            for idx in range(len(letters) - 1):
                curr = letters[idx]
                nxt = letters[idx + 1]
                self.transition_counts[curr][nxt] += 1
                self.transition_totals[curr] += 1

            last_state = letters[-1]
            self.transition_counts[last_state][END_STATE] += 1
            self.transition_totals[last_state] += 1

            for idx, state_char in enumerate(letters):
                if idx >= len(typed):
                    # No observation recorded for missing character when smoothing is disabled.
                    continue
                typed_char = typed[idx]
                self.emission_counts[state_char][typed_char] += 1
                self.emission_totals[state_char] += 1

        self.state_set.discard("\n")
        self.states = sorted(self.state_set)

        self._finalize_probabilities()

    def _finalize_probabilities(self) -> None:
        self.start_log_probs = {}
        self.transition_log_probs = defaultdict(dict)
        self.emission_log_probs = defaultdict(dict)

        if not self.states:
            return

        for state in self.states:
            count = self.start_counts.get(state, 0)
            if self.start_total == 0 or count == 0:
                self.start_log_probs[state] = float("-inf")
            else:
                prob = count / self.start_total
                self.start_log_probs[state] = math.log(prob)

        for prev_state in self.states:
            total = self.transition_totals.get(prev_state, 0)
            if total == 0:
                continue
            for next_state, count in self.transition_counts[prev_state].items():
                if count == 0:
                    continue
                prob = count / total
                self.transition_log_probs[prev_state][next_state] = math.log(prob)

        for state in self.states:
            total = self.emission_totals.get(state, 0)
            if total == 0:
                continue
            for observation, count in self.emission_counts[state].items():
                if count == 0:
                    continue
                prob = count / total
                self.emission_log_probs[state][observation] = math.log(prob)

    def _log_start_prob(self, state: str) -> float:
        if self.start_log_probs:
            return self.start_log_probs.get(state, float("-inf"))

        total = self.start_total
        count = self.start_counts.get(state, 0)
        if total == 0 or count == 0:
            return float("-inf")
        prob = count / total
        return math.log(prob) if prob > 0 else float("-inf")

    def _log_transition_prob(self, prev_state: str, next_state: str) -> float:
        cached = self.transition_log_probs.get(prev_state)
        if cached and next_state in cached:
            return cached[next_state]

        total = self.transition_totals.get(prev_state, 0)
        count = self.transition_counts[prev_state].get(next_state, 0)
        if total == 0 or count == 0:
            return float("-inf")
        prob = count / total
        return math.log(prob) if prob > 0 else float("-inf")

    def _log_emission_prob(self, state: str, observation: str) -> float:
        cached = self.emission_log_probs.get(state)
        if cached and observation in cached:
            return cached[observation]

        count = self.emission_counts[state].get(observation, 0)
        total = self.emission_totals.get(state, 0)
        if total == 0 or count == 0:
            return float("-inf")
        prob = count / total
        return math.log(prob) if prob > 0 else float("-inf")

    def transition_log_distribution(self, state: str) -> List[Tuple[str, float]]:
        """Return sorted log-probabilities for transitions leaving `state`."""
        items = list(self.transition_log_probs.get(state, {}).items())
        return sorted(items, key=lambda kv: kv[1], reverse=True)

    def emission_log_distribution(self, state: str) -> List[Tuple[str, float]]:
        """Return sorted log-probabilities for emissions produced by `state`."""
        items = list(self.emission_log_probs.get(state, {}).items())
        return sorted(items, key=lambda kv: kv[1], reverse=True)

    def decode_word(self, word: str) -> str:
        # Viterbi algorithm
        if not word or not self.states:
            return word

        observed = word.lower()
        length = len(observed)
        viterbi: List[Dict[str, float]] = [{} for _ in range(length)]
        path: Dict[str, List[str]] = {}

        first_obs = observed[0]
        for state in self.states:
            score = self._log_start_prob(state) + self._log_emission_prob(state, first_obs)
            viterbi[0][state] = score
            path[state] = [state]

        for index in range(1, length):
            obs_char = observed[index]
            new_path: Dict[str, List[str]] = {}
            for curr_state in self.states:
                emission_score = self._log_emission_prob(curr_state, obs_char)
                best_prev: Optional[str] = None
                best_score = float("-inf")
                for prev_state in self.states:
                    if prev_state not in path:
                        continue
                    prev_score = viterbi[index - 1].get(prev_state)
                    if prev_score is None:
                        continue
                    transition_score = self._log_transition_prob(prev_state, curr_state)
                    score = prev_score + transition_score
                    if score > best_score:
                        best_score = score
                        best_prev = prev_state
                if best_prev is not None:
                    viterbi[index][curr_state] = best_score + emission_score
                    new_path[curr_state] = path[best_prev] + [curr_state]
                else:
                    viterbi[index][curr_state] = float("-inf")
            path = new_path

        final_state: Optional[str] = None
        final_score = float("-inf")
        for state in self.states:
            state_score = viterbi[length - 1].get(state)
            if state_score is None:
                continue
            total_score = state_score + self._log_transition_prob(state, END_STATE)
            if total_score > final_score:
                final_score = total_score
                final_state = state

        if final_state is None:
            return word

        decoded = "".join(path[final_state])
        if word.isupper():
            return decoded.upper()
        if word[0].isupper():
            return decoded.capitalize()
        return decoded

    def decode_text(self, text: str) -> str:
        words = text.split()
        corrected_words = [self.decode_word(word) for word in words]
        return " ".join(corrected_words)


def main() -> None:
    pairs = load_training_pairs("aspell.txt")
    corrector = SpellCheckerHMM()
    corrector.train(pairs)

    print("Hidden Markov Model spell checker.")
    print("Enter text to correct (Ctrl-D to exit).")
    print("Commands: :trans <state> for transitions, :emit <state> for emissions.")

    def _print_distribution(kind: str, items: List[Tuple[str, float]], state: str) -> None:
        if not items:
            print(f"No {kind} data recorded for '{state}'.")
            return
        arrow = "->" if kind == "transition" else "~>"
        for symbol, log_prob in items:
            prob = math.exp(log_prob) if log_prob != float("-inf") else 0.0
            print(f"{state} {arrow} {symbol}: log={log_prob:.4f}, prob={prob:.6f}")

    try:
        while True:
            line = input("> ")
            if line.startswith(":trans"):
                parts = line.split()
                if len(parts) >= 2:
                    state = parts[1].lower()
                    _print_distribution("transition", corrector.transition_log_distribution(state), state)
                else:
                    print("Usage: :trans <state>")
                continue
            if line.startswith(":emit"):
                parts = line.split()
                if len(parts) >= 2:
                    state = parts[1].lower()
                    _print_distribution("emission", corrector.emission_log_distribution(state), state)
                else:
                    print("Usage: :emit <state>")
                continue

            corrected = corrector.decode_text(line)
            print(corrected)
    except EOFError:
        print()


if __name__ == "__main__":
    main()
