# STEP 01 - Prepare career text for embedding.
# Inputs: occupation_data.csv, essential_skills.csv, task_statements.csv in
# DATA/Career Profile Files. Output: processed_career_profiles.csv there.
# Setup: Python 3.12; install requirements.txt. First use downloads the tokenizer.
# Run from the repository root:
#   python SCRIPTS/01_prepare_career_profiles.py --data-dir "DATA/Career Profile Files"
# The explicit path is needed because --data-dir otherwise defaults to DATA.
# Existing output is overwritten. Run step 02 next.
# All described occupations remain eligible; missing skills/tasks are not imputed.

"""Build complete and token-budgeted career profiles with MiniLM's tokenizer.

Run from the repository root with the --data-dir command in the header.
Install dependencies from requirements.txt. No embeddings are generated here.
`full_profile_text` preserves the original top-10-skills/all-tasks profile.
`profile_text` selects from ALL ranked skills and whole tasks, aiming for
245-250 tokens including special tokens. Short source records are not padded.
"""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


# Shared join key and fixed tokenizer version keep preparation consistent with encoding.
CODE = "O*NET-SOC Code"
DATA_DIR = Path(__file__).resolve().parents[1] / "DATA"
MISSING = {"", "nan", "none", "null", "n/a", "na"}
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
TOKEN_BUDGET = 250


def render_profile(title, description, skills, tasks):
    # Omit empty sections so missing source content does not become model input.
    sections = [("Occupation", title), ("Description", description),
                ("Skills", "; ".join(skills)), ("Tasks", " ".join(tasks))]
    return " ".join(f"{label}: {value}" for label, value in sections if value)


def shorten_profile(title, description, skills, tasks, tokenizer, budget=TOKEN_BUDGET):
    """Deterministic selection of whole source fields; never token-slice text.

    Retain the title and description, reserve tasks before allocating skills,
    then favor tasks with Core status, broad respondent support, and vocabulary
    not already covered by selected tasks. This is a transparent heuristic,
    not a semantic summary or a guarantee of optimal information coverage.
    """
    selected_skills, selected_tasks = [], []

    def text_for(skill_ids, task_ids):
        return render_profile(title, description,
                              [skills[i] for i in sorted(skill_ids)],
                              [tasks[i]["text"] for i in sorted(task_ids)])

    def count(skill_ids, task_ids):
        return len(tokenizer.encode(text_for(skill_ids, task_ids),
                                    add_special_tokens=True, truncation=False))

    # Count actual model tokens, including labels and special tokens, before selection.
    base_count = count([], [])
    if base_count > budget:
        raise ValueError(f"{title}: title and description exceed {budget} tokens; "
                         "manual sentence-level revision is required")
    words = [set(re.findall(r"[a-z]{3,}", task["text"].lower())) for task in tasks]

    def task_order():
        covered = set().union(*(words[i] for i in selected_tasks))
        def priority(i):
            # Support is a respondent count; division by 100 is a weighting heuristic,
            # not a conversion to a response percentage. Source order breaks ties.
            novelty = len(words[i] - covered) / max(1, len(words[i]))
            return (-tasks[i]["core"], -(tasks[i]["support"] / 100 + novelty), i)
        return sorted((i for i in range(len(tasks)) if i not in selected_tasks), key=priority)

    # Reserve up to two whole tasks, ideally alongside the five best skills.
    # If necessary, reduce the skill reservation so tasks are represented.
    for _ in range(2):
        chosen = None
        for reserve in range(min(5, len(skills)), -1, -1):
            chosen = next((i for i in task_order()
                           if count(list(range(reserve)), selected_tasks + [i]) <= budget), None)
            if chosen is not None:
                break
        if chosen is not None:
            selected_tasks.append(chosen)

    # Spend roughly 40% of the post-description budget on ranked skills.
    # Remaining space goes to tasks, with skills filling small final gaps.
    skill_allowance = max(0, int((budget - base_count) * 0.4))
    for i in range(len(skills)):
        proposed = selected_skills + [i]
        if (count(proposed, []) - base_count <= skill_allowance
                and count(proposed, selected_tasks) <= budget):
            selected_skills.append(i)
    while True:
        chosen = next((i for i in task_order()
                       if count(selected_skills, selected_tasks + [i]) <= budget), None)
        if chosen is None:
            break
        selected_tasks.append(chosen)
    for i in range(len(skills)):
        if i not in selected_skills and count(selected_skills + [i], selected_tasks) <= budget:
            selected_skills.append(i)

    # Whole statements leave gaps. Try exchanging one selected task for a
    # longer complete task, without reducing the number of Core tasks. Keep
    # the skill selection fixed and stop once the 245-250 target is reached.
    while count(selected_skills, selected_tasks) < 245:
        best_count = count(selected_skills, selected_tasks)
        best_tasks = None
        for incoming in task_order():
            for outgoing in reversed(selected_tasks):
                if tasks[incoming]["core"] < tasks[outgoing]["core"]:
                    continue
                proposed = [i for i in selected_tasks if i != outgoing] + [incoming]
                proposed_count = count(selected_skills, proposed)
                if best_count < proposed_count <= budget:
                    best_count, best_tasks = proposed_count, proposed
        if best_tasks is None:
            break
        selected_tasks = best_tasks
    # A swap may change boundary tokenization; fill any remaining whole units.
    for i in range(len(skills)):
        if i not in selected_skills and count(selected_skills + [i], selected_tasks) <= budget:
            selected_skills.append(i)
    for i in task_order():
        if count(selected_skills, selected_tasks + [i]) <= budget:
            selected_tasks.append(i)

    if tasks and not selected_tasks:
        raise ValueError(f"{title}: no complete task fits beside its title and description")
    text = text_for(selected_skills, selected_tasks)
    return text, count(selected_skills, selected_tasks), selected_skills, selected_tasks


def clean(value):
    """Remove missing-value placeholders and normalize whitespace."""
    value = " ".join((value or "").split())
    return "" if value.casefold() in MISSING else value


def read_rows(path, required):
    # Accept UTF-8 with or without a BOM; fail early if a required column is absent.
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            yield {key: clean(value) for key, value in row.items() if key}


def prepare_profiles(data_dir, top_skills=10, tokenizer=None):
    if top_skills < 0:
        raise ValueError("top_skills must be nonnegative")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)

    # The occupation table defines the output population. Repeated rows fill
    # missing fields without multiplying profiles during the one-to-many joins.
    occupations = {}
    for row in read_rows(data_dir / "occupation_data.csv", [CODE, "Title", "Description"]):
        code = row[CODE]
        if not code:
            continue
        occupation = occupations.setdefault(code, {"Title": "", "Description": ""})
        for field in occupation:
            if not occupation[field]:
                occupation[field] = row[field]

    # Use Importance ratings only; Level scores measure a different quantity.
    # Suppression/relevance flags remain in the source but are not filtered here.
    skills = defaultdict(dict)
    for row in read_rows(
        data_dir / "essential_skills.csv",
        [CODE, "Element Name", "Scale ID", "Data Value"],
    ):
        code, name = row[CODE], row["Element Name"]
        if code not in occupations or not name or row["Scale ID"].upper() != "IM":
            continue
        try:
            score = float(row["Data Value"])
        except ValueError:
            continue
        if not math.isfinite(score):
            continue
        key = name.casefold()
        previous = skills[code].get(key)
        # For repeated skill names, retain the highest available Importance.
        if previous is None or score > previous[1]:
            skills[code][key] = (name, score)

    # Group unique task text by occupation; missing respondent counts get zero
    # selection weight. Prefer Core status and stronger support for duplicates.
    tasks = defaultdict(dict)
    for row in read_rows(data_dir / "task_statements.csv", [CODE, "Task"]):
        code, task = row[CODE], row["Task"]
        if code in occupations and task:
            try:
                support = float(row.get("Incumbents Responding", ""))
                if not math.isfinite(support):
                    support = 0.0
            except ValueError:
                support = 0.0
            candidate = {"text": task, "core": row.get("Task Type", "").casefold() == "core",
                         "support": support}
            previous = tasks[code].get(task.casefold())
            if previous is None or (candidate["core"], support) > (previous["core"], previous["support"]):
                tasks[code][task.casefold()] = candidate

    # Sort career codes to make output order stable and retain the full source
    # profile beside the shortened text so selections can be inspected later.
    profiles = []
    for code, occupation in sorted(occupations.items()):
        ranked = sorted(skills[code].values(), key=lambda item: (-item[1], item[0].casefold()))
        all_skills = [name for name, _ in ranked]
        original_skills = all_skills[:top_skills] if top_skills else all_skills
        task_records = list(tasks[code].values())
        all_tasks = [task["text"] for task in task_records]
        full_profile_text = render_profile(occupation["Title"], occupation["Description"],
                                           original_skills, all_tasks)
        profile_text, token_count, skill_ids, task_ids = shorten_profile(
            occupation["Title"], occupation["Description"], all_skills, task_records, tokenizer)
        # An entirely empty record cannot provide a useful embedding input.
        if profile_text:
            profiles.append({
                CODE: code,
                **occupation,
                "Skills": "; ".join(original_skills),
                "Tasks": " ".join(all_tasks),
                "full_profile_text": full_profile_text,
                "profile_text": profile_text,
                "token_count": token_count,
                "selected_skills": "; ".join(all_skills[i] for i in sorted(skill_ids)),
                "selected_tasks": " ".join(all_tasks[i] for i in sorted(task_ids)),
            })
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output", type=Path, help="Default: DATA directory/processed_career_profiles.csv")
    parser.add_argument("--top-skills", type=int, default=10,
                        help="Skills in preserved full profile only; shortened profiles consider all ranked skills")
    args = parser.parse_args()
    if args.top_skills < 0:
        parser.error("--top-skills must be nonnegative")
    profiles = prepare_profiles(args.data_dir, args.top_skills)
    output = args.output or args.data_dir / "processed_career_profiles.csv"
    # Write a named-column CSV consumed by step 02; create its folder if needed.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=[CODE, "Title", "Description", "Skills", "Tasks",
                               "full_profile_text", "profile_text", "token_count", "selected_skills", "selected_tasks"])
        writer.writeheader()
        writer.writerows(profiles)
    print(f"Saved {len(profiles):,} occupation profiles to {output}")


if __name__ == "__main__":
    main()
