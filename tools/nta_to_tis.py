import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ProcessedAutomaton:
    id: int
    comments: str
    locations: List[str]
    transitions: List[str]
    valuation: str


def _extract_preceding_comment_block(content: str, start_idx: int) -> str:
    """Return the contiguous block of comment lines (and blank lines between them)
    immediately preceding content[start_idx:].

    A line is treated as a comment if it starts with '#' after optional leading spaces.
    """
    lines = content[:start_idx].splitlines(keepends=True)

    block: List[str] = []
    seen_comment = False

    i = len(lines) - 1
    while i >= 0:
        line = lines[i]
        stripped = line.strip()

        if stripped == "":
            if seen_comment:
                block.append(line)
            i -= 1
            continue

        if line.lstrip().startswith("#"):
            seen_comment = True
            block.append(line)
            i -= 1
            continue

        break

    if not seen_comment:
        return ""

    # Preserve original formatting; ensure we end with a single blank line separator.
    text = "".join(reversed(block)).rstrip() + "\n\n"
    return text


def nta_to_tis(nta_path: str, tis_path: str) -> None:
    with open(nta_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # 1) Pobieranie liczby zegarów
    clock_match = re.search(r"^\s*clocks\s+(\+?\d+)\s*$", content, re.MULTILINE)
    num_clocks = int(clock_match.group(1)) if clock_match else 0
    clock_names = [f"x{i}" for i in range(num_clocks)]

    # 2) Parsowanie automatów + zbieranie resetów dla action_id
    #    Szukamy bloków:
    #      automaton #ID
    #        ...
    #      end #automaton ID
    automata_iter = re.finditer(
        r"^\s*automaton\s+#(\d+)\s*\n(.*?)\n\s*end\s+#automaton\s+\1\s*$",
        content,
        re.DOTALL | re.MULTILINE,
    )

    action_resets: Dict[int, Tuple[str, ...]] = {}  # action_id -> tuple(clock names)
    processed_automata: List[ProcessedAutomaton] = []

    for m in automata_iter:
        a_id = int(m.group(1))
        body = m.group(2)

        auto_comments = _extract_preceding_comment_block(content, m.start())

        # Jeśli komentarz ma nagłówek w stylu "# Automaton #<id>", dopasuj numerację do przesuniętych ID.
        if auto_comments:
            auto_comments = re.sub(
                r"(^\s*#\s*Automaton\s+#)\d+",
                rf"\g<1>{a_id + 1}",
                auto_comments,
                flags=re.MULTILINE,
            )

        # Lokacje i wyceny kopiujemy (bez ingerencji).
        locations = re.findall(r"location\s+\d+\s+end", body)
        valuation_block = re.search(r"valuation.*?\nend", body, re.DOTALL)

        # Tranzycje: z .tis usuwamy linie 'reset ...' z automatów, bo resety trzymamy w środowisku (#0).
        transitions = re.findall(r"transition\s+(\d+)\s+(\d+)\s+(\d+)(.*?)\nend", body, re.DOTALL)

        current_auto_transitions: List[str] = []
        for src, dst, act_id_str, extra in transitions:
            act_id = int(act_id_str)

            resets = re.search(r"^\s*reset\s+([\w\s]+)\s*$", extra, re.MULTILINE)
            if resets:
                clock_list = tuple(sorted(resets.group(1).split()))
                action_resets[act_id] = clock_list

            # Zachowujemy warunki czasowe / ograniczenia (wszystko poza 'reset ...')
            conditions: List[str] = []
            for line in extra.splitlines():
                s = line.strip()
                if not s:
                    continue
                if s.startswith("reset "):
                    continue
                conditions.append(s)

            trans_text = f"transition {src} {dst} {act_id}\n"
            for cond in conditions:
                trans_text += f"{cond}\n"
            trans_text += "end"
            current_auto_transitions.append(trans_text)

        processed_automata.append(
            ProcessedAutomaton(
                id=a_id + 1,  # przesuwamy ID, bo #0 zajmie środowisko
                comments=auto_comments,
                locations=locations,
                transitions=current_auto_transitions,
                valuation=valuation_block.group(0) if valuation_block else "",
            )
        )

    # 3) Generowanie pliku .tis
    with open(tis_path, "w", encoding="utf-8") as f:
        f.write("network\n\n")
        f.write("#environment\n\n")
        f.write("automaton \n#0\n\n")
        f.write(f"clocks {' '.join(clock_names)} \n\n")
        f.write("location 0\nend\n\n")

        # Automaton #0: resety zebrane po action_id
        for act_id in sorted(action_resets.keys()):
            resets = action_resets[act_id]
            f.write(f"transition 0 0 {act_id}\n")
            if resets:
                f.write(f"reset {' '.join(resets)}\n")
            f.write("end\n\n")

        f.write("valuation\n")
        f.write("end\n\n")
        f.write("end\n")
        f.write("#" + "-" * 33 + "\n")

        # Reszta automatów (z zachowaniem komentarzy poprzedzających każdy automat w .nta)
        for auto in processed_automata:
            if auto.comments:
                f.write(auto.comments)

            f.write(f"automaton \n#{auto.id}\n\n")

            for loc in auto.locations:
                f.write(f"{loc}\n\n")

            for trans in auto.transitions:
                f.write(f"{trans}\n\n")

            if auto.valuation:
                f.write(f"{auto.valuation}\n")

            f.write("end\n")
            f.write("#" + "-" * 33 + "\n")
        f.write("end\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Użycie: python nta_to_tis.py <wejscie.nta> <wyjscie.tis>")
        raise SystemExit(2)

    nta_to_tis(sys.argv[1], sys.argv[2])
