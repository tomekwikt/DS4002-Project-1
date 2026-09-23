"""Create extractive, token-budgeted resume + interest profiles.

Run: .venv/Scripts/python.exe SCRIPTS/03_prepare_user_profiles.py
Uses the pinned MiniLM tokenizer, not an LLM; no qualifications are inferred.
Original CSV fields are retained verbatim. See README.md for selection rules.
"""

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DATA_DIR = Path(__file__).resolve().parents[1] / "DATA"
STEM = "career_match_20_resumes_with_generated_interests"
BUDGET = 250
SECTION_PATTERN = re.compile(
    r"\b(Education and Training|Professional Experience|Professional Summary|"
    r"Professional Affiliations|Additional Information|Activities and Honors|"
    r"Areas of Expertise|Technical Skills|Computer Skills|Skill Highlights|"
    r"Core Strengths|Career Overview|Work Experience|Work History|"
    r"Accomplishments|Qualifications|Certifications|Publications|Presentations|"
    r"Community Service|Affiliations|Achievements|Highlights|Experience|Employment|Education|"
    r"Summary|Profile|Skills|Interests|Languages|Portfolio)\b"
)
EDUCATION = {"Education", "Education and Training"}
EXPERIENCE = {"Professional Experience", "Work Experience", "Work History", "Experience", "Employment", "Accomplishments"}
SKILLS = {"Skills", "Technical Skills", "Computer Skills", "Skill Highlights", "Core Strengths", "Highlights", "Areas of Expertise", "Qualifications", "Certifications", "Languages"}
SUMMARY = {"Summary", "Professional Summary", "Career Overview", "Profile"}
STOP = set("a an the and or of to in for with on as at by from my i me our their that this is are was were have has had it its be being been work working worked skills experience ability strong excellent professional company city state".split())
ACTION = r"(?:Analyzed|Analyze|Designed|Design|Developed|Created|Managed|Maintained|Coordinated|Implemented|Prepared|Provided|Conducted|Assisted|Responsible|Performed|Supported|Taught|Trained|Led|Built|Monitored|Communicated|Researched|Tested|Oversaw|Organized|Facilitated|Evaluated|Improved|Reviewed|Utilized|Collaborated|Helped)"
DEGREE = r"(?:Master(?:s|'s)?(?: of [A-Za-z]+)?|Bachelor(?:s|'s)?(?: of [A-Za-z]+)?|Associate(?: of [A-Za-z]+ [A-Za-z]+| Degree)?|BSEE|BBA|MBA|AAS Degree|M\.Ed|B\.S|BA|BS|AA|Certificate in [A-Za-z ]+(?= :))"


def normalize(text):
    """Formatting cleanup only; preserve original spelling and qualifications."""
    text = re.sub(r"[\u200b\ufeff]", "", text)
    text = re.sub(r"[\u2022\u25cf\u25e6]", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+|\b[\w.+-]+@[\w.-]+\.[A-Za-z]+\b", " ", text)
    text = re.sub(r"\bCompany Name\b", " ", text)
    text = re.sub(r"[－–—-]?\s*\bCity\s*,\s*State\b(?:\s*,\s*(?:United States|USA|US))?", " ", text)
    text = re.sub(r"_{2,}", " ", text)
    text = " ".join(text.split()).strip(" ,;-－–—")
    # Remove adjacent repeated phrases (common in exported education fields).
    for _ in range(3):
        text = re.sub(r"\b(\S+(?:\s+\S+){1,15})\s+\1(?=\s|$)", r"\1", text, flags=re.I)
    return text


def words(text):
    return set(re.findall(r"[a-z][a-z0-9+#-]*", text.lower())) - STOP


def unique(units):
    result, seen = [], set()
    for text in units:
        text = normalize(text)
        key = re.sub(r"\W+", " ", text.casefold()).strip()
        if key and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def split_list(text):
    """Split list separators without breaking parenthesized qualifications."""
    depth, start, parts = 0, 0, []
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif char in ",;" and not depth:
            parts.append(text[start:index])
            start = index + 1
    parts.append(text[start:])
    return parts


def split_units(text, category):
    # Protect common abbreviations before splitting complete sentences.
    protected = re.sub(r"\b(?:[A-Za-z]\.){2,}|\b(?:Dr|Mr|Ms|St|Univ|M\.Ed|B\.S)\.",
                       lambda m: m[0].replace(".", "\u2024"), text)
    units = re.split(r"(?<=[.!?])\s+(?=[A-Z])|(?<=[a-z][.!?])(?=[A-Z])|[\r\n]+", protected)
    output = []
    for unit in units:
        unit = unit.replace("\u2024", ".")
        if category == "Skills":
            # Comma/semicolon lists supply complete skill phrases.
            for part in split_list(unit):
                output.extend(re.split(r"\s+(?=(?:Adobe|Microsoft|Proficient|Strong|Excellent|"
                                       r"Knowledgeable|Qualified|Certified|CPR|HIPAA|Trained|"
                                       r"Python|SQL|Windows|Linux|Ability|Skilled|Experienced)\b)", part))
        elif category == "Education":
            # Keep institutions, dates, degrees, and completion qualifiers
            # together: degree names can occur BEFORE or AFTER institutions.
            output.extend(re.split(r";", unit))
        elif len(unit.split()) > 55:
            # Flattened bullet lists have no sentence terminators. Split only
            # at capitalized action-led statements, never at a token cutoff.
            output.extend(re.split(r"\s+(?=" + ACTION + r"\b)", unit))
        else:
            output.append(unit)
    return unique(output)


def extract_resume(raw):
    text = normalize(raw)
    matches = []
    for match in SECTION_PATTERN.finditer(text):
        before = text[max(0, match.start() - 22):match.start()]
        after = text[match.end():match.end() + 35]
        if match[0] == "Education" and (
            re.search(r"(?:Special|Elementary|Secondary|General|Physical)\s+$", before)
            or re.match(r"\s+(?:and Safety|endorsement|Elementary|GPA|\d{4}\s+Southern)", after)
        ):
            continue
        # Lowercase prose such as 'problem Solving Skills' is not a header.
        if match[0] == "Skills" and re.search(r"(?:Solving|Organizational|Telephone|Communication)\s+$", before):
            continue
        if match[0] == "Skills" and re.search(r"Clinical\s+$", before):
            continue
        matches.append(match)
    title = text[:matches[0].start()].strip() if matches else ""
    # Exported resumes occasionally put a person's name after the role title.
    role = re.match(r"[A-Z][A-Z0-9/&()-]*(?:\s+[A-Z0-9/&()-]+)*(?=\s|$)", title)
    if role:
        title = role[0].strip()
    sections = defaultdict(list)
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = text[match.end():end].strip()
        category = ("Education" if match[0] in EDUCATION else
                    "Experience" if match[0] in EXPERIENCE else
                    "Skills" if match[0] in SKILLS else
                    "Summary" if match[0] in SUMMARY else None)
        if category:
            sections[category].extend(split_units(value, category))
    for category in list(sections):
        sections[category] = unique(sections[category])
    # Reject isolated export debris, not substantive technical abbreviations.
    debris = {"dec", "view", "works", "etc", "etc.", "increase", "class", "office",
              "clients", "client", "phone", "written", "government", "features", "forms",
              "managers", "printer", "fax machine", "works great with others"}
    sections["Skills"] = [u for u in sections.get("Skills", []) if u.casefold().strip(".") not in debris]
    sections["Education"] = [u for u in sections.get("Education", [])
                             if len(words(u)) >= 3 and not re.match(r"(?:While|I |This |Also )", u)]
    if not any(sections.values()):
        sections["Experience"] = split_units(text, "Experience")
    return title, sections


def build_profile(resume, interests, tokenizer):
    title, sections = extract_resume(resume)
    interests = normalize(interests)
    if not interests or not normalize(resume):
        raise ValueError("Both resume and interests must be nonempty")
    selected = {category: [] for category in ("Education", "Experience", "Skills", "Summary")}
    interest_words = words(interests)
    token_cache = {}

    def render(selection):
        parts = [f"Background: {title}"] if title else []
        for category, units in selection.items():
            if units:
                joined = ("; ".join(units) if category == "Skills" else
                          " ".join(unit if unit.endswith((".", "!", "?")) else unit + "." for unit in units))
                parts.append(f"{category}: " + joined)
        parts.append(f"Interests: {interests}")
        return " ".join(parts)

    def count(selection):
        text = render(selection)
        if text not in token_cache:
            token_cache[text] = len(tokenizer.encode(text, add_special_tokens=True, truncation=False))
        return token_cache[text]

    def with_unit(category, unit):
        return {key: values + [unit] if key == category else values[:]
                for key, values in selected.items()}

    def rank(category, unit):
        terms = words(unit)
        covered = words(" ".join(x for units in selected.values() for x in units))
        overlap = len(terms & interest_words)
        novelty = len(terms - covered) / max(1, len(terms))
        # Concrete actions/outcomes outrank generic self-descriptions.
        action = bool(re.search(r"\b" + ACTION + r"\b", unit))
        degree = bool(re.search(DEGREE, unit)) if category == "Education" else False
        # Long undelimited skill lists should not displace concrete experience.
        long_list_penalty = max(0, len(unit.split()) - 20) * .2 if category == "Skills" else 0
        specific = .8 * bool(re.search(r"\d", unit)) if category == "Experience" else 0
        return 3 * overlap + 3 * action + 4 * degree + 2 * novelty + specific - long_list_penalty

    def options(category):
        return sorted((unit for unit in sections.get(category, [])
                       if unit not in selected[category]),
                      key=lambda unit: (-rank(category, unit), sections[category].index(unit)))

    if count(selected) > BUDGET:
        raise ValueError("Title and complete interests exceed the budget; revise source interests")
    # Protect each resume category from being crowded out by other sections.
    # Each seed leaves 28 tokens for the remaining available categories.
    categories = [c for c in ("Education", "Experience", "Skills") if sections.get(c)]
    for index, category in enumerate(categories):
        cap = BUDGET - 28 * (len(categories) - index - 1)
        fits = [unit for unit in options(category) if count(with_unit(category, unit)) <= cap]
        if fits:
            selected = with_unit(category, fits[0])

    # Add distinct, relevant source units. Discourage near-duplicate sentences
    # and repeated skill phrases already expressed in selected experience.
    while True:
        candidates = []
        existing = [words(x) for units in selected.values() for x in units]
        for category in ("Experience", "Education", "Skills", "Summary"):
            for unit in options(category):
                terms = words(unit)
                existing_text = " ".join(x for units in selected.values() for x in units).casefold()
                if (not terms or unit.casefold().rstrip(".") in existing_text
                        or (category == "Skills" and terms <= words(existing_text))
                        or any(len(terms & prior) / max(1, len(terms | prior)) >= .65
                               or (len(terms) >= 6 and len(prior) >= 6
                                   and len(terms & prior) / min(len(terms), len(prior)) >= .85)
                               for prior in existing)):
                    continue
                proposed = with_unit(category, unit)
                n = count(proposed)
                if n <= BUDGET:
                    # Prefer underrepresented categories, then relevance and
                    # information per token. Stable source order breaks ties.
                    balance = 3 / (1 + len(selected[category]))
                    cost = max(1, n - count(selected))
                    candidates.append((rank(category, unit) + balance + min(len(terms), 20) / cost,
                                       category, unit, proposed))
        if not candidates:
            break
        selected = max(candidates, key=lambda x: x[0])[3]
    # If whole units leave a small gap, replace a selected unit with a more
    # informative, no-lower-ranked unit in the SAME category. Coverage stays
    # intact, and no text is sliced merely to reach the desired length.
    if count(selected) < 245:
        swaps = []
        for category in selected:
            for incoming in options(category):
                for outgoing in selected[category]:
                    if rank(category, incoming) < rank(category, outgoing):
                        continue
                    retained = [unit for key, units in selected.items() for unit in units
                                if key != category or unit != outgoing]
                    terms = words(incoming)
                    if any(len(terms & words(unit)) / max(1, len(terms | words(unit))) >= .65
                           for unit in retained):
                        continue
                    proposed = {key: [incoming if unit == outgoing else unit for unit in units]
                                if key == category else units[:] for key, units in selected.items()}
                    n = count(proposed)
                    if 245 <= n <= BUDGET:
                        swaps.append((rank(category, incoming), n, proposed))
        if swaps:
            selected = max(swaps, key=lambda item: (item[0], item[1]))[2]
    if not any(selected.values()):
        raise ValueError("No complete resume unit fits beside the interests")
    for category in categories:
        if not selected[category]:
            raise ValueError(f"No complete {category.lower()} unit fits; review source formatting")
    return render(selected), count(selected), title, selected


def find_input(data_dir):
    # Prefer the user's exact filename. Accept the existing unsuffixed copy.
    for filename in (STEM + "(2).csv", STEM + ".csv"):
        matches = sorted(data_dir.rglob(filename))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError("Multiple input files found; select one with --input")
    raise FileNotFoundError(f"No {STEM} CSV under {data_dir}; provide --input")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "processed_user_profiles.csv")
    args = parser.parse_args()
    source = args.input or find_input(DATA_DIR)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    with source.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"Resume_str", "Generated_Interests"} <= set(reader.fieldnames or []):
            raise ValueError("Input requires Resume_str and Generated_Interests columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Input contains no people")
    results, seen = [], defaultdict(int)
    for index, row in enumerate(rows, 1):
        resume, interests = row["Resume_str"] or "", row["Generated_Interests"] or ""
        digest = hashlib.sha256((resume + "\0" + interests).encode("utf-8")).hexdigest()[:16]
        seen[digest] += 1
        user_id = f"user_{digest}_{seen[digest]:02d}"
        try:
            text, n, title, selected = build_profile(resume, interests, tokenizer)
        except ValueError as error:
            raise ValueError(f"Source row {index}: {error}") from error
        if n > BUDGET:
            raise ValueError(f"Source row {index} exceeds token budget")
        results.append({"user_id": user_id, "source_row": index,
                        "original_resume": resume, "original_interests": interests,
                        "profile_text": text, "token_count": n,
                        "selected_education": " | ".join(selected["Education"]),
                        "selected_experience": " | ".join(selected["Experience"]),
                        "selected_skills": " | ".join(selected["Skills"]),
                        "selected_summary": " | ".join(selected["Summary"])})
    counts = [len(ids) for ids in tokenizer([r["profile_text"] for r in results],
              add_special_tokens=True, padding=False, truncation=False)["input_ids"]]
    assert counts == [r["token_count"] for r in results]
    assert max(counts) <= BUDGET < 256
    assert len({r["user_id"] for r in results}) == len(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    stats = {"model": MODEL_NAME, "revision": MODEL_REVISION, "users": len(results),
             "average_tokens": statistics.mean(counts), "median_tokens": statistics.median(counts),
             "minimum_tokens": min(counts), "maximum_tokens": max(counts),
             "profiles_in_245_to_250_range": sum(245 <= n <= 250 for n in counts),
             "percentage_truncated": 0.0}
    args.output.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Input: {source}\nSaved {len(results)} profiles to {args.output}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
