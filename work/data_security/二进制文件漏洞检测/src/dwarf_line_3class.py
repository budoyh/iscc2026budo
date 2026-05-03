from __future__ import annotations

import io
from collections import Counter

import pandas as pd
import pefile
from elftools.dwarf.dwarfinfo import DWARFInfo, DebugSectionDescriptor, DwarfConfig
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from common import BINARY_DIR, RAW, REPORT, SUBMISSIONS


TARGET_CWES = {"CWE-121", "CWE-122", "CWE-126"}


def _attr(die, name: str):
    if die is None:
        return None
    attr = die.attributes.get(name)
    if attr is None:
        return None
    value = attr.value
    if isinstance(value, bytes):
        return value.decode(errors="ignore")
    return value


def _debug_sections(binary_id: str) -> dict[str, bytes]:
    raw = (BINARY_DIR / f"{binary_id}.exe").read_bytes()
    pe = pefile.PE(data=raw, fast_load=False)
    ptr = pe.FILE_HEADER.PointerToSymbolTable
    count = pe.FILE_HEADER.NumberOfSymbols
    strtab_offset = ptr + count * 18
    strtab_size = int.from_bytes(raw[strtab_offset : strtab_offset + 4], "little") if strtab_offset + 4 <= len(raw) else 0
    strtab = raw[strtab_offset + 4 : strtab_offset + strtab_size]

    sections: dict[str, bytes] = {}
    for section in pe.sections:
        raw_name = section.Name.rstrip(b"\x00")
        if raw_name.startswith(b"/"):
            offset = int(raw_name[1:]) - 4
            end = strtab.find(b"\x00", offset)
            name = strtab[offset:end].decode(errors="ignore")
        else:
            name = raw_name.decode(errors="ignore")
        data = raw[section.PointerToRawData : section.PointerToRawData + section.SizeOfRawData] if section.SizeOfRawData else b""
        sections[name] = data[: section.Misc_VirtualSize or len(data)]
    return sections


def _dwarf_info(binary_id: str) -> DWARFInfo:
    sections = _debug_sections(binary_id)

    def desc(name: str) -> DebugSectionDescriptor:
        data = sections.get(name, b"")
        return DebugSectionDescriptor(io.BytesIO(data), name, 0, len(data), 0)

    return DWARFInfo(
        DwarfConfig(little_endian=True, machine_arch="x64", default_address_size=8),
        desc(".debug_info"),
        desc(".debug_aranges"),
        desc(".debug_abbrev"),
        desc(".debug_frame"),
        desc(".eh_frame"),
        desc(".debug_str"),
        None,
        None,
        desc(".debug_line"),
        None,
        None,
        None,
        None,
        desc(".debug_line_str"),
        None,
        None,
        None,
        None,
        None,
    )


def _deref(die, die_map: dict[int, object]):
    seen: set[int] = set()
    while die is not None and die.offset not in seen and die.tag in {
        "DW_TAG_typedef",
        "DW_TAG_const_type",
        "DW_TAG_volatile_type",
    }:
        seen.add(die.offset)
        die = die_map.get(_attr(die, "DW_AT_type"))
    return die


def _simple_type(die, die_map: dict[int, object]) -> str:
    die = _deref(die, die_map)
    if die is None:
        return "void"
    if die.tag == "DW_TAG_pointer_type":
        target = _deref(die_map.get(_attr(die, "DW_AT_type")), die_map)
        return "*" + (str(_attr(target, "DW_AT_name")) if target is not None else "void")
    if die.tag == "DW_TAG_array_type":
        elem = _deref(die_map.get(_attr(die, "DW_AT_type")), die_map)
        elem_name = str(_attr(elem, "DW_AT_name")) if elem is not None else "?"
        dims = []
        for child in die.iter_children():
            if child.tag == "DW_TAG_subrange_type":
                upper = _attr(child, "DW_AT_upper_bound")
                count = _attr(child, "DW_AT_count")
                dims.append(str(count if count is not None else (upper + 1 if isinstance(upper, int) else upper)))
        return elem_name + "[" + "][".join(dims) + "]"
    return f"{_attr(die, 'DW_AT_name')}[{_attr(die, 'DW_AT_byte_size')}]"


def extract_dwarf_line_features(binary_id: str) -> dict[str, object]:
    features: dict[str, object] = {
        "binary_id": binary_id,
        "entry_line": -1,
        "high_pc": -1,
        "var_count": 0,
        "var_lines": "",
        "var_line_min": -1,
        "var_line_delta_min": -1,
        "ptr_sig": "",
        "struct_size_sig": "",
        "member_sig": "",
        "has_j": 0,
        "has_structCharVoid": 0,
        "char_first_type": "",
        "line_seq": "",
        "line_seq_compact": "",
        "line_delta_seq": "",
        "line_count": 0,
        "line_unique_count": 0,
    }
    try:
        dwarf = _dwarf_info(binary_id)
        for cu in dwarf.iter_CUs():
            top_name = str(_attr(cu.get_top_DIE(), "DW_AT_name") or "")
            if "source_sanitized.c" not in top_name:
                continue
            die_map = {die.offset: die for die in cu.iter_DIEs()}
            for die in cu.iter_DIEs():
                if die.tag != "DW_TAG_subprogram" or _attr(die, "DW_AT_name") != "entry_bad":
                    continue
                entry_line = int(_attr(die, "DW_AT_decl_line") or -1)
                low_pc = int(_attr(die, "DW_AT_low_pc") or -1)
                high_pc = int(_attr(die, "DW_AT_high_pc") or -1)
                features["entry_line"] = entry_line
                features["high_pc"] = high_pc
                if low_pc >= 0 and high_pc > 0:
                    try:
                        line_program = dwarf.line_program_for_CU(cu)
                        lines: list[int] = []
                        for entry in line_program.get_entries():
                            if entry.state is None or entry.state.end_sequence:
                                continue
                            address = int(entry.state.address)
                            if low_pc <= address < low_pc + high_pc:
                                lines.append(int(entry.state.line or -1))
                        compact: list[int] = []
                        for line in lines:
                            if not compact or compact[-1] != line:
                                compact.append(line)
                        features["line_seq"] = "|".join(str(line) for line in lines)
                        features["line_seq_compact"] = "|".join(str(line) for line in compact)
                        features["line_delta_seq"] = "|".join(str(line - entry_line) for line in compact)
                        features["line_count"] = len(lines)
                        features["line_unique_count"] = len(set(lines))
                    except Exception:
                        pass
                var_lines: list[int] = []
                ptr_flags: list[str] = []
                struct_sizes: list[str] = []
                member_parts: list[str] = []
                char_types: list[str] = []

                def visit(node) -> None:
                    for child in node.iter_children():
                        if child.tag == "DW_TAG_variable":
                            name = str(_attr(child, "DW_AT_name") or "")
                            line = int(_attr(child, "DW_AT_decl_line") or -1)
                            var_lines.append(line)
                            if name == "j":
                                features["has_j"] = 1
                            type_die = die_map.get(_attr(child, "DW_AT_type"))
                            base = _deref(type_die, die_map)
                            ptr = 0
                            if base is not None and base.tag == "DW_TAG_pointer_type":
                                ptr = 1
                                base = _deref(die_map.get(_attr(base, "DW_AT_type")), die_map)
                            ptr_flags.append(f"{name}:{ptr}")
                            if base is not None and base.tag == "DW_TAG_structure_type":
                                if name == "structCharVoid":
                                    features["has_structCharVoid"] = 1
                                struct_sizes.append(f"{name}:{_attr(base, 'DW_AT_byte_size')}")
                                for member in base.iter_children():
                                    if member.tag != "DW_TAG_member":
                                        continue
                                    member_type = _simple_type(die_map.get(_attr(member, "DW_AT_type")), die_map)
                                    part = f"{_attr(member, 'DW_AT_name')}@{_attr(member, 'DW_AT_data_member_location')}:{member_type}"
                                    member_parts.append(part)
                                    if str(_attr(member, "DW_AT_name") or "") == "charFirst":
                                        char_types.append(member_type)
                        else:
                            visit(child)

                visit(die)
                features["var_count"] = len(var_lines)
                features["var_lines"] = "|".join(str(v) for v in var_lines)
                features["var_line_min"] = min(var_lines) if var_lines else -1
                features["var_line_delta_min"] = (min(var_lines) - entry_line) if var_lines and entry_line >= 0 else -1
                features["ptr_sig"] = "|".join(ptr_flags)
                features["struct_size_sig"] = "|".join(struct_sizes)
                features["member_sig"] = "|".join(sorted(set(member_parts)))
                features["char_first_type"] = "|".join(sorted(set(char_types)))
                return features
    except Exception as exc:
        features["member_sig"] = f"ERR:{type(exc).__name__}"
    return features


def main() -> None:
    train = pd.read_csv(RAW / "train.csv", keep_default_na=False)
    train_tri = train[(train["label"].astype(int) == 1) & train["cwe_id"].isin(TARGET_CWES)].copy()
    candidate_ids = pd.read_csv(REPORT / "s17b.csv.diff.csv", keep_default_na=False)["binary_id"].tolist()
    all_ids = train_tri["binary_id"].tolist() + candidate_ids
    feature_rows = [extract_dwarf_line_features(binary_id) for binary_id in all_ids]
    features = pd.DataFrame(feature_rows)
    train_features = features.iloc[: len(train_tri)].copy()
    train_features["cwe_id"] = train_tri["cwe_id"].values
    test_features = features.iloc[len(train_tri) :].copy()

    numeric = [
        "entry_line",
        "high_pc",
        "var_count",
        "var_line_min",
        "var_line_delta_min",
        "has_j",
        "has_structCharVoid",
        "line_count",
        "line_unique_count",
    ]
    categorical = [
        "var_lines",
        "ptr_sig",
        "struct_size_sig",
        "member_sig",
        "char_first_type",
        "line_seq_compact",
        "line_delta_seq",
    ]
    pre = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=1), categorical),
        ]
    )
    models = {
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200,
            random_state=20260503,
            class_weight="balanced",
            min_samples_leaf=1,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=20260503,
            class_weight="balanced",
            min_samples_leaf=1,
            n_jobs=-1,
        ),
    }
    x = train_features[numeric + categorical]
    y = train_features["cwe_id"].astype(str).to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260503)
    print(f"train={len(train_features)} class_counts={Counter(y)}")
    predictions = []
    for name, model in models.items():
        pipe = make_pipeline(pre, model)
        cv_pred = cross_val_predict(pipe, x, y, cv=cv, n_jobs=1)
        print(f"{name} cv_macro={f1_score(y, cv_pred, average='macro'):.12f}")
        pipe.fit(x, y)
        pred = pipe.predict(test_features[numeric + categorical])
        proba = pipe.predict_proba(test_features[numeric + categorical])
        classes = list(pipe.named_steps[type(model).__name__.lower()].classes_)
        for i, binary_id in enumerate(candidate_ids):
            predictions.append(
                {
                    "model": name,
                    "binary_id": binary_id,
                    "dwarf_pred": pred[i],
                    "dwarf_prob": float(proba[i].max()),
                    **{f"prob_{cls}": float(proba[i, classes.index(cls)]) for cls in classes},
                }
            )

    pred_frame = pd.DataFrame(predictions)
    diff = pd.read_csv(REPORT / "s17b.csv.diff.csv", keep_default_na=False)
    out = diff.merge(test_features, on="binary_id", how="left").merge(pred_frame, on="binary_id", how="left")
    out.to_csv(REPORT / "s17b_dwarf_line_3class.csv", index=False, encoding="utf-8")
    print(
        out[
            [
                "model",
                "binary_id",
                "sid",
                "old_cwe",
                "new_cwe",
                "entry_line",
                "high_pc",
                "var_lines",
                "line_seq_compact",
                "ptr_sig",
                "dwarf_pred",
                "dwarf_prob",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
