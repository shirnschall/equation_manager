##################################################
# Equation Manager for RQ Calculators
##################################################

"""
Equation Manager module

This module provides a generic system for managing and solving symbolic systems of equations
using SymPy. It supports dynamic equation assembly based on templates and provided input values.

The core idea is to keep the equation solver generic so it can be reused across different
domains (e.g. aerodynamics, propulsion, structures) by adding domain-specific equation
templates in separate files (like ``everything_aero.py``).

Features:
- Variables are indexed (e.g. v1, v2, rho3) to support multiple instances
- Templates use placeholders ``{{i}}`` and ``{{j}}`` for single/multi-index equations
- Equations are only added when required input variables are present
- Solutions are presented with LaTeX formatting in Jupyter environments
- Fallback solving strategy drops selected equations if the full system is overconstrained
"""

# from sympy import symbols, Eq, solve, exp, S, latex, simplify, sympify
# import numpy as np
from copy import deepcopy
from functools import reduce
import re
from collections import defaultdict
from itertools import combinations

def _load_sympy():
    global symbols, Eq, solve, exp, S, latex, simplify, sympify
    from sympy import symbols, Eq, solve, exp, S, latex, simplify, sympify

def in_jupyter():
    try:
        # Most common & reliable way (works in classic notebook, JupyterLab, Colab, VSCode notebooks...)
        from IPython import get_ipython
        ipy = get_ipython()
        if ipy is None:
            return False
        # Check if we're in an interactive backend that supports rich display
        shell = ipy.__class__.__name__
        return 'Terminal' not in shell and 'Interactive' in shell or 'ZMQ' in shell
    except ImportError:
        return False
    
if in_jupyter():
    from IPython.display import Math, Latex, display, Markdown
    import plotly.graph_objects as go
    
    # Optional: you can also set plotly to notebook mode here if desired
    # import plotly.io as pio
    # pio.renderers.default = 'notebook'   # or 'jupyterlab', 'colab', etc.
    
    print("Jupyter environment detected → rich display & plotly imported")
else:
    print("Not in Jupyter → skipping IPython & plotly imports")

_load_sympy() #lazy load sympy

class Equation_Template:
    """Container for a single equation template used by :class:`Equation_Manager`.

    :param equations_to_add: List of equation strings defining the equations.
                             Placeholders ``{{i}}`` and ``{{j}}`` are replaced with concrete indices. Lhs and rhs are separated by a single '='.
    :type equations_to_add: list[str]

    :param relevant_vars: List of ``(base_name, description)`` tuples for variables
                          involved in these equations.
    :type relevant_vars: list[tuple[str, str]]

    :param vars_to_check: List of ``([var_names], min_count)`` tuples. The template is
                          only activated if at least ``min_count`` indexed instances exist
                          for the listed base variable names.
    :type vars_to_check: list[tuple[list[str], int]]

    :param name: Human-readable name of the template (used in reports and pretty-print)
    :type name: str
    """

    def __init__(self, equations_to_add: list, relevant_vars: list, vars_to_check: list, name: str):
        # TODO: check for {{i}} AND {{j}} to determine multiindex or not
        self.equations_to_add = equations_to_add
        self.relevant_vars = relevant_vars
        self.vars_to_check = vars_to_check
        self.name = name


class Equation_Manager:
    """Generic symbolic equation system manager and solver.

    Dynamically assembles and solves systems of equations based on provided input
    values and registered equation templates.

    Main workflow:
        1. Add templates via :meth:`add_equation_template`
        2. Call :meth:`solve` with input values and (optional) assumptions
        3. Equations are assembled → solved → pretty-printed (in Jupyter)

    :ivar variables: dict[str, list[sympy.Symbol]] — grouped by base name
    :ivar symbol_db: dict[str, sympy.Symbol] — flat lookup "v1" → Symbol('v1')
    :ivar equations: list[sympy.Eq] — collected equations
    :ivar equation_names: list[str] — corresponding names (may be empty)
    :ivar equation_templates: list[Equation_Template] — registered templates
    """

    def __init__(self):
        self._reset()
        self.reset_equation_templates()

    def _reset(self):
        """Reset internal state (variables, equations, values) — keeps templates."""
        self.variables = {}
        self.variable_indices = {}      # deprecated — prefer symbol_db
        self.symbol_db = {}
        self.equations = []
        self.equation_names = []
        self.input_eqs = []

    def reset_equation_templates(self):
        """Clear all registered equation templates."""
        self.equation_templates = []

    def _add_vars(self, variable_names: list, index: int):
        """Create symbolic variables for given base names + index if not already present.

        :param variable_names: List of (base_name, description) tuples
        :param index: Integer suffix (e.g. 1 → v1, rho1)
        """
        for vn in variable_names:
            k = vn[0]
            var_str = f"{k}{index}"
            if var_str not in self.symbol_db:
                if k not in self.variables:
                    self.variables[k] = []
                sym = symbols(var_str, real=True)
                self.variables[k].append(sym)
                self.symbol_db[var_str] = sym
                if k not in self.variable_indices:
                    self.variable_indices[k] = []
                self.variable_indices[k].append(index)

    def _n_vars_given(self, n: int, relevant_vars: list, index: int) -> bool:
        """Check whether at least ``n`` of the listed variables exist with given index.

        :param n: minimum number of matching indexed variables required
        :param relevant_vars: list of base variable names to check
        :param index: index suffix to look for
        :return: True if condition is satisfied
        """
        count = sum(1 for k in relevant_vars if f"{k}{index}" in self.symbol_db)
        return count >= n

    def _add_equation_unique(self, eqn) -> int:
        """Add equation only if an equivalent one does not already exist.

        :param eqn: sympy Eq object
        :return: 1 if added, 0 if already present
        """
        eqn_str = str(eqn).replace(" ", "")
        if eqn_str not in [str(eq).replace(" ", "") for eq in self.equations]:
            self.equations.append(eqn)
            return 1
        return 0

    def _pretty_print_eqns(self, header_txt, equation_names, equations):
        """Display equations in LaTeX format in Jupyter.

        :param header_txt: Section title
        :param equation_names: list of equation labels (may be empty)
        :param equations: list of sympy Eq objects
        """
        display(Math(rf"\Large \textbf{{{header_txt}}}"))

        if not equations:
            display(Math(r"\textit{No equations}"))
            return

        content = r" \\[4pt] ".join(
            rf"\text{{{str(name).replace('_', ' ')}}} & {latex(eq)}"
            for name, eq in zip(equation_names, equations)
        )

        display(Math(r"\begin{array}{rl}" + content + r"\end{array}"))
        display(Markdown("---"))

    def _pretty_print_sol(self, sol, looking_for=[]):
        """Pretty-print solution dictionary in LaTeX (Jupyter).

        :param sol: dict[sympy.Symbol, value]
        :param looking_for: optional list of variable name substrings to highlight first
        """
        display(Math(r"\Large \textbf{Solution}"))

        if not sol:
            display(Math(r"\textit{No solution}"))
            return

        lines = []
        # prioritized variables first
        for k in sol:
            if looking_for and any(lf in str(k) for lf in looking_for):
                lines.append(rf"\bbox[yellow]{{{latex(k)}}} & \bbox[yellow]{{{latex(sol[k])}}} \\[4pt]")

        # then the rest
        for k in sol:
            if not looking_for or not any(lf in str(k) for lf in looking_for):
                lines.append(rf"{latex(k)} & {latex(sol[k])} \\[4pt] ")

        display(Math(r"\begin{array}{rl}" + "".join(lines) + r"\end{array}"))
        display(Markdown("---"))

    def add_equation_template(self, equations_to_add: list, relevant_vars: list,
                              vars_to_check: list, name: str = ""):
        r"""Register a new equation template.

        Templates are only activated when enough indexed instances of required
        variables are present (controlled by ``vars_to_check``).

        :param equations_to_add: list of ``lhs_str=rhs_str`` strings using ``{{i}}``/``{{j}}``
        :param relevant_vars: list of ``(base_name, description)`` tuples
        :param vars_to_check: list of ``([var_names], min_count)`` activation conditions
        :param name: optional template identifier (shown in pretty-print output)
        """
        self.equation_templates.append(
            Equation_Template(equations_to_add, relevant_vars, vars_to_check, name)
        )

    def assemble_equations(self):
        """Iteratively assemble concrete equations from all active templates.

        Continues until no new equations are added (fixed-point iteration).
        """

        

        #equations from templates
        while True:
            equations_added = 0

            for eqt in self.equation_templates:
                # Collect all known indices from relevant variables
                index_sets = [
                    set(self.variable_indices.get(var_name, []))
                    for var_name, _ in eqt.relevant_vars
                    if var_name in self.variable_indices
                ]
                combined_indices = reduce(lambda x, y: x | y, index_sets, set())

                for eqn in eqt.equations_to_add:
                    multiindex = "{{j}}" in eqn.split("=")[0] or "{{j}}" in eqn.split("=")[1] or not combined_indices
                    # print(f"{multiindex}: Processing template equation: {eqn}", flush=True)

                    indices_j = deepcopy(combined_indices) if multiindex else \
                                [max(combined_indices) + 1] if combined_indices else [1]

                    for i in sorted(combined_indices):
                        for j in sorted(indices_j):
                            if i >= j:
                                continue

                            # Check activation conditions
                            if all(self._n_vars_given(cnt, vars_, i) or
                                   self._n_vars_given(cnt, vars_, j)
                                   for vars_, cnt in eqt.vars_to_check):

                                if multiindex:
                                    self._add_vars(eqt.relevant_vars, j)
                                    print(f"Added vars for j={j}: {eqt.relevant_vars}", flush=True)
                                self._add_vars(eqt.relevant_vars, i)
                                print(f"Added vars for i={i}: {eqt.relevant_vars}", flush=True)

                                lhs_str = eqn.split("=")[0].replace("{{i}}", str(i)).replace("{{j}}", str(j))
                                rhs_str = eqn.split("=")[1].replace("{{i}}", str(i)).replace("{{j}}", str(j))

                                lhs = sympify(lhs_str, locals=self.symbol_db)
                                rhs = simplify(sympify(rhs_str, locals=self.symbol_db))

                                if self._add_equation_unique(Eq(lhs, rhs)):
                                    equations_added += 1
                                    if eqt.name:
                                        self.equation_names.append(f"{eqt.name}_{i}")

            if equations_added == 0:
                break

        # equations from input
        for eq in self.input_eqs:
            lhs, rhs = eq.split('=')
            self._add_equation_unique(
                    Eq(sympify(lhs, locals=self.symbol_db), sympify(rhs, locals=self.symbol_db)))
        # print(f"equations from input: {self.equations}")
        

    def _add_input(self, input_equations: list):
        """Parse and register input values, creating variables as needed.

        :param input_dict: ``{"v1": 33.3, "rho2": 1.225, ...}``
        """
        pattern = r'\b[A-Za-z][A-Za-z_]*(_0|\d+)?\b'

        for eq in input_equations:
            self.input_eqs.append(eq)
            for x in eq.split('='):
                s_ind = 0
                while m := re.search(pattern, str(x)[s_ind:]):
                    s_ind += m.end()
                    self._add_vars([(m.group()[:-len(m.group(1))], "")], int(m.group(1)))

    def ngapp_vars(self):
        return_sting = ""

        # return_sting += r"\(\text{\textbf{Variables:}}\\\\\)"

        all_vars = {}
        for tmpl in self.equation_templates:
            for var, desc in tmpl.relevant_vars:
                if var not in all_vars:
                    all_vars[var] = desc

        sorted_vars = sorted(all_vars.items())

        # lines = [r"\(\begin{aligned}"]
        # for var, desc in sorted_vars:
        #     safe_var = var.replace('_', r'\_')
        #     safe_desc = desc.replace('_', r'\_')
        #     lines.append(rf"\text{{{safe_var}}} &\quad\ldots \text{{{safe_desc}}} \\")
        # lines.append(r"\end{aligned}\)")

        # return_sting +="".join(lines)

        name_lines = []
        desc_lines = []
        for var, desc in sorted_vars:
            name_lines.append(f"{var}\n")
            desc_lines.append(f"{desc}\n")

        names = "".join(name_lines)
        desc = "".join(desc_lines)

        return names, desc

    def ngapp_equations(self):
        return_sting = ""

        return_sting += r"\(\text{\textbf{Equations:}}\\\)"

        eq_lines = []
        local_symbols = {}
        for tmpl in self.equation_templates:
            for var, _ in tmpl.relevant_vars:
                if var not in local_symbols:
                    local_symbols[var] = symbols(var)

        for tmpl in self.equation_templates:
            for eq in tmpl.equations_to_add:
                lhs_raw, rhs_raw = eq.split('=')
                lhs = sympify(lhs_raw.replace("{{i}}","").replace("{{j}}",""), locals=local_symbols)
                rhs = sympify(rhs_raw.replace("{{i}}","").replace("{{j}}",""), locals=local_symbols)
                name = tmpl.name or "—"
                eq_lines.append(
                    rf"\text{{{name.replace('_', ' ')}}} & {latex(Eq(lhs, rhs))} \\[4pt]"
                )

        if eq_lines:
            return_sting += r"\(\begin{aligned}" + "".join(eq_lines) + r"\end{aligned}\)"
        else:
            return_sting += r"\(\textit{No templates loaded yet.}\)"

        return return_sting


    def help(self):
        """Print help text and display available variables & equations in Jupyter."""
        print("Equation Manager Help:")
        print("  • add_equation_template()    – register new equation sets")
        print("  • solve(values, assumptions, looking_for)   – solve system")
        print("  • Input keys must use indices:  v1, rho2, C_L3, ...")
        print("  • Solutions shown with LaTeX in Jupyter notebooks\n")

        display(Markdown("---"))

        print("Available variables:")

        all_vars = {}
        for tmpl in self.equation_templates:
            for var, desc in tmpl.relevant_vars:
                if var not in all_vars:
                    all_vars[var] = desc

        sorted_vars = sorted(all_vars.items())

        lines = [r"\begin{array}{r@{$\quad\longrightarrow\quad$}l}"]
        for var, desc in sorted_vars:
            safe_var = var.replace('_', r'\_')
            safe_desc = desc.replace('_', r'\_')
            lines.append(rf"\text{{{safe_var}}} & \text{{{safe_desc}}} \\\\")

        lines.append(r"\end{array}")
        display(Math("".join(lines)))

        display(Markdown("---"))

        print("Available equation templates:")

        eq_lines = []
        local_symbols = {}
        for tmpl in self.equation_templates:
            for var, _ in tmpl.relevant_vars:
                if var not in local_symbols:
                    local_symbols[var] = symbols(var)

        for tmpl in self.equation_templates:
            for eq in tmpl.equations_to_add:
                lhs_raw, rhs_raw = eq.split('=')
                lhs = sympify(lhs_raw.replace("{{i}}","").replace("{{j}}",""), locals=local_symbols)
                rhs = sympify(rhs_raw.replace("{{i}}","").replace("{{j}}",""), locals=local_symbols)
                name = tmpl.name or "—"
                eq_lines.append(
                    rf"\text{{{name.replace('_', ' ')}}} & {latex(Eq(lhs, rhs))} \\\\[4pt]"
                )

        if eq_lines:
            display(Math(r"\begin{array}{rl}" + "".join(eq_lines) + r"\end{array}"))
        else:
            display(Math(r"\textit{No templates loaded yet.}"))

    def _solve(self, assumptions: list, looking_for: list, max_drop_cnt: int = 3):
        """Solve the system — fall back to dropping assumption-related equations if needed.

        :param assumptions: fixed values that may overconstrain the system
        :param looking_for: variables to prioritize keeping in solution
        :param max_drop_cnt: maximum number of equations to remove in one try
        :return: (solutions list, used equations, used equation names)
        """
        vars_to_solve = list(self.symbol_db.values())

        def extract_symbols(eq_str):
            pattern = r'\b[A-Za-z][A-Za-z0-9_]*\b'
            return set(re.findall(pattern, eq_str))
        
        assumption_set = set()
        for assump in assumptions:
            assumption_set.update(extract_symbols(assump))

        print(f"assumption symbols: {assumption_set}")

        solution = solve(
            self.equations,
            vars_to_solve,
            dict=True,
            domain=S.Reals
        )

        if solution:
            return solution, self.equations, self.equation_names

        # Fallback: try dropping equations containing assumption variables
        assumption_eq_indices = [
            i for i, eq in enumerate(self.equations)
            if any(str(self.symbol_db.get(k, "")) in str(eq) for k in assumption_set)
        ]

        for keep_looking in [False, True]:
            for drop_count in range(1, max_drop_cnt + 1):
                print(f"Trying to drop {drop_count} assumption-related equations...")

                for indices_to_drop in combinations(assumption_eq_indices, drop_count):
                    if keep_looking:
                        indices_to_drop = tuple(
                            i for i in indices_to_drop
                            if any(v in str(self.equations[i]) for v in looking_for)
                        )
                        if not indices_to_drop:
                            continue
                    else:
                        # When not keeping looking_for → skip if any looking_for eq would be dropped
                        if any(any(v in str(self.equations[i]) for v in looking_for)
                               for i in indices_to_drop):
                            continue

                    reduced = [eq for i, eq in enumerate(self.equations) if i not in indices_to_drop]

                    sol = solve(reduced, vars_to_solve, dict=True, domain=S.Reals)
                    if sol:
                        print(f"→ Solution found after dropping indices {indices_to_drop}")
                        used_eqs = [eq for i, eq in enumerate(self.equations) if i not in indices_to_drop]
                        used_names = [n for i, n in enumerate(self.equation_names) if i not in indices_to_drop]
                        return sol, used_eqs, used_names

        print("No solution found even after dropping equations.")
        return [], self.equations, self.equation_names

    def solve(self, values: list, assumptions: list, looking_for: list,
              pretty_print: bool = True):
        """Main entry point — solve the equation system with given inputs.

        :param values: known variable values (e.g. ``{"v1": 33.0, "rho1": 1.225}``)
        :param assumptions: additional fixed values (may be relaxed if no solution)
        :param looking_for: variables to highlight / prioritize in output
        :param pretty_print: whether to display LaTeX output in Jupyter
        :return: (list of solutions, list of final equations)
        """
        self._reset()
        self._add_input(values)
        self._add_input(assumptions)
        print(f"symbolic variables created: {self.symbol_db}")
        self.assemble_equations()

        print("Solving... ", end="")
        print(f"self.equations: {self.equations}")
        print(f"self.equation_names: {self.equation_names}")
        print(f"assumptions: {assumptions}")
        print(f"looking_for: {looking_for}")

        # self._pretty_print_eqns("Matching Equations", self.equation_names, self.equations)

        solutions, eqs_used, names_used = self._solve(assumptions, looking_for)

        print("Done.")

        if pretty_print:
            if not solutions:
                print("No solution found.")
            else:
                print(f"Found {len(solutions)} solution(s). Showing first one:")
                self._pretty_print_eqns("Equations actually used:", names_used, eqs_used)
                self._pretty_print_sol(solutions[0], looking_for)

        return solutions, eqs_used, names_used