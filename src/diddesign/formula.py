from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FACTOR_RE = re.compile(r"^factor\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)$")


@dataclass(frozen=True)
class CovariateTermSpec:
    """A single atomic covariate component (column reference)."""
    source: str
    categorical: bool


@dataclass(frozen=True)
class InteractionSpec:
    """Parsed covariate expression: single term or interaction of multiple terms."""
    terms: tuple[CovariateTermSpec, ...]
    # len == 1: plain covariate
    # len >= 2: interaction (e.g. x1:x2)


def _is_bool_like(value: Any) -> bool:
    return isinstance(value, bool) or (
        type(value).__module__ == "numpy"
        and type(value).__name__ in {"bool", "bool_"}
    )


def _split_top_level(expression: str, delimiter: str) -> tuple[str, ...]:
    if len(delimiter) != 1:
        raise ValueError("delimiter must be a single character.")

    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for character in expression:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Invalid formula (rhs)")
        if character == delimiter and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(character)

    if depth != 0:
        raise ValueError("Invalid formula (rhs)")

    parts.append("".join(current).strip())
    return tuple(parts)


def _split_terms(expression: str) -> tuple[str, ...]:
    if not expression:
        return ()
    terms = tuple(term.strip() for term in _split_top_level(expression, "+"))
    if any(not term for term in terms):
        raise ValueError("Invalid formula (rhs)")
    return terms


def _parse_covariate_variables(terms: tuple[str, ...]) -> tuple[str, ...]:
    variables: list[str] = []
    # Track which variables appeared as standalone plain main effects
    plain_main_effects: set[str] = set()
    for term in terms:
        try:
            specs = parse_covariate_expression(term)
        except (TypeError, ValueError):
            raise ValueError("Invalid formula (rhs)") from None
        is_star_expansion = len(specs) > 1
        for spec in specs:
            if len(spec.terms) == 1:
                # Plain main effect term
                src = spec.terms[0].source
                if src in plain_main_effects and not is_star_expansion:
                    raise ValueError("Invalid formula (rhs)")
                if not is_star_expansion:
                    plain_main_effects.add(src)
                if src not in variables:
                    variables.append(src)
            else:
                # Interaction term: add all component sources (no dup error)
                for component in spec.terms:
                    if component.source not in variables:
                        variables.append(component.source)
    return tuple(variables)


def parse_covariate_term(term: str) -> tuple[str, bool]:
    """Parse a single atomic covariate term (no interaction operators)."""
    if not isinstance(term, str):
        raise TypeError("covariates must be a sequence of column names or factor(...) terms.")

    stripped = term.strip()
    if not stripped:
        raise ValueError("covariates cannot contain blank entries.")

    factor_match = _FACTOR_RE.fullmatch(stripped)
    if factor_match is not None:
        return factor_match.group(1), True

    if _IDENTIFIER_RE.fullmatch(stripped):
        return stripped, False

    if stripped.startswith("factor(") and stripped.endswith(")"):
        source = stripped[7:-1].strip()
        if not source:
            raise ValueError("factor() covariate terms must name a column.")
    raise ValueError("covariates must contain only column names or factor(column) terms.")


def _parse_atomic_component(token: str) -> CovariateTermSpec:
    """Parse a single component within an interaction expression."""
    source, categorical = parse_covariate_term(token)
    return CovariateTermSpec(source=source, categorical=categorical)


def parse_covariate_expression(term: str) -> list[InteractionSpec]:
    """Parse a covariate expression that may contain `*` or `:` operators.

    Returns a list of InteractionSpec objects.  For plain terms this is a
    single-element list; for `*` expressions the list contains the main
    effects and the interaction (e.g. ``a*b`` -> [a, b, a:b]).
    """
    if not isinstance(term, str):
        raise TypeError("covariates must be a sequence of column names or factor(...) terms.")
    stripped = term.strip()
    if not stripped:
        raise ValueError("covariates cannot contain blank entries.")

    # Check for `*` operator (full crossing = main effects + interaction)
    star_parts = _split_top_level(stripped, "*")
    if len(star_parts) > 1:
        # e.g. "x1*factor(x2)" -> main effects x1, factor(x2), plus interaction x1:factor(x2)
        components = [_parse_atomic_component(p) for p in star_parts]
        result: list[InteractionSpec] = []
        # Add each main effect
        for comp in components:
            result.append(InteractionSpec(terms=(comp,)))
        # Add the full interaction
        result.append(InteractionSpec(terms=tuple(components)))
        return result

    # Check for `:` operator (pure interaction)
    colon_parts = _split_top_level(stripped, ":")
    if len(colon_parts) > 1:
        components = [_parse_atomic_component(p) for p in colon_parts]
        return [InteractionSpec(terms=tuple(components))]

    # Plain single term
    comp = _parse_atomic_component(stripped)
    return [InteractionSpec(terms=(comp,))]


def _require_identifier(value: str | None, *, field_name: str, allow_none: bool = False) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a valid column identifier.")
    return value


def _require_distinct_formula_roles(**roles: str | None) -> None:
    seen: dict[str, str] = {}
    for role, column in roles.items():
        if column is None:
            continue
        previous_role = seen.get(column)
        if previous_role is not None:
            raise ValueError(f"{role} must be distinct from {previous_role}.")
        seen[column] = role


@dataclass(frozen=True)
class DidFormulaSpec:
    fm_did: tuple[str, str]
    fm_covar: str | None
    var_outcome: str
    var_treat: str
    var_post: str | None
    var_covars: tuple[str, ...]
    covariate_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.var_outcome, field_name="var_outcome")
        _require_identifier(self.var_treat, field_name="var_treat")
        _require_identifier(self.var_post, field_name="var_post", allow_none=True)
        _require_distinct_formula_roles(
            var_outcome=self.var_outcome,
            var_treat=self.var_treat,
            var_post=self.var_post,
        )

        if not isinstance(self.covariate_terms, tuple):
            raise TypeError("covariate_terms must be a tuple of formula covariate terms.")
        role_columns = {self.var_outcome, self.var_treat}
        if self.var_post is not None:
            role_columns.add(self.var_post)
        seen_covariates: set[str] = set()
        expected_var_covars: list[str] = []
        plain_main_effects: set[str] = set()
        for term in self.covariate_terms:
            specs = parse_covariate_expression(term)
            is_star_expansion = len(specs) > 1
            for spec in specs:
                for component in spec.terms:
                    if component.source in role_columns:
                        raise ValueError("covariate_terms must not reuse outcome, treatment, or post columns.")
                if len(spec.terms) == 1:
                    src = spec.terms[0].source
                    if src in plain_main_effects and not is_star_expansion:
                        raise ValueError("covariate_terms must not contain duplicate column names.")
                    if not is_star_expansion:
                        plain_main_effects.add(src)
                    if src not in seen_covariates:
                        seen_covariates.add(src)
                        expected_var_covars.append(src)
                else:
                    for component in spec.terms:
                        if component.source not in seen_covariates:
                            seen_covariates.add(component.source)
                            expected_var_covars.append(component.source)
        if self.var_covars != tuple(expected_var_covars):
            raise ValueError("var_covars must match the unique variables in covariate_terms.")

        expected_fm_covar = " + ".join(self.covariate_terms) if self.covariate_terms else None
        if self.fm_covar != expected_fm_covar:
            raise ValueError("fm_covar must match covariate_terms.")

        did_terms = ["Gi", "It", "Gi:It"]
        did_terms.extend(self.covariate_terms)
        expected_did_rhs = " + ".join(did_terms)
        expected_fm_did = (
            f"outcome ~ {expected_did_rhs}",
            f"outcome_delta ~ {expected_did_rhs}",
        )
        if self.fm_did != expected_fm_did:
            raise ValueError("fm_did must match the canonical DID formula.")

    def as_did_kwargs(self) -> dict[str, object]:
        return {
            "outcome": self.var_outcome,
            "treatment": self.var_treat,
            "post": self.var_post,
            "covariates": self.covariate_terms or None,
        }


def did_formula(formula: str, is_panel: bool) -> DidFormulaSpec:
    if not isinstance(formula, str):
        raise TypeError("formula must be a string.")
    if not _is_bool_like(is_panel):
        raise TypeError("is_panel must be a boolean.")
    is_panel = bool(is_panel)

    formula_parts = _split_top_level(formula.strip(), "~")
    if len(formula_parts) != 2:
        raise ValueError("Invalid formula (rhs)")

    lhs, rhs = formula_parts
    lhs_terms = _split_terms(lhs)
    if len(lhs_terms) != 1 or not _IDENTIFIER_RE.fullmatch(lhs_terms[0]):
        raise ValueError("Invalid formula (lhs)")

    rhs_sections = _split_top_level(rhs, "|")
    if len(rhs_sections) not in {1, 2}:
        raise ValueError("Invalid formula (rhs)")
    if len(rhs_sections) == 2 and not rhs_sections[1].strip():
        raise ValueError("Invalid formula (rhs)")

    main_terms = _split_terms(rhs_sections[0])
    covariate_terms = _split_terms(rhs_sections[1]) if len(rhs_sections) == 2 else ()
    if is_panel:
        if len(main_terms) != 1:
            raise ValueError("Invalid formula (rhs)")
        var_treat = main_terms[0]
        var_post = None
    else:
        if len(main_terms) != 2:
            raise ValueError(
                "Repeated cross-section formulas must specify treatment + post in that order."
            )
        var_treat, var_post = main_terms

    if not _IDENTIFIER_RE.fullmatch(var_treat) or (
        var_post is not None and not _IDENTIFIER_RE.fullmatch(var_post)
    ):
        raise ValueError("Invalid formula (rhs)")

    fm_covar = " + ".join(covariate_terms) if covariate_terms else None
    var_covars = _parse_covariate_variables(covariate_terms)
    did_terms = ["Gi", "It", "Gi:It"]
    did_terms.extend(covariate_terms)
    did_rhs = " + ".join(did_terms)

    return DidFormulaSpec(
        fm_did=(
            f"outcome ~ {did_rhs}",
            f"outcome_delta ~ {did_rhs}",
        ),
        fm_covar=fm_covar,
        var_outcome=lhs_terms[0],
        var_treat=var_treat,
        var_post=var_post,
        var_covars=var_covars,
        covariate_terms=covariate_terms,
    )


__all__ = [
    "CovariateTermSpec",
    "InteractionSpec",
    "DidFormulaSpec",
    "did_formula",
    "parse_covariate_term",
    "parse_covariate_expression",
]
