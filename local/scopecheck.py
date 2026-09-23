import ast, sys, builtins

def child_nodes(node):
    """Walk a function body WITHOUT descending into nested function scopes."""
    for n in ast.iter_child_nodes(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        yield n
        for c in child_nodes(n):
            yield c

def bound_names(node):
    """Names bound within this scope (not nested ones)."""
    out = set()
    for n in child_nodes(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)          # 'except X as e'
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out.update(n.names)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                out.add((al.asname or al.name).split('.')[0])
    return out

def params_of(fn):
    a = fn.args
    out = {x.arg for x in list(a.args) + list(a.kwonlyargs)
           + list(getattr(a, 'posonlyargs', []))}
    if a.vararg: out.add(a.vararg.arg)
    if a.kwarg: out.add(a.kwarg.arg)
    return out

def analyze(fn, inherited, path, bad):
    scope = set(inherited) | params_of(fn) | bound_names(fn)
    for n in child_nodes(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            if n.id not in scope and not hasattr(builtins, n.id):
                bad.append((path, getattr(fn, 'name', '<lambda>'),
                            n.lineno, n.id))
    for n in ast.walk(fn):
        if n is not fn and isinstance(n, (ast.FunctionDef, ast.Lambda)):
            analyze(n, scope, path, bad)

def check(path):
    tree = ast.parse(open(path).read())
    top = bound_names(tree) | {n.name for n in ast.walk(tree)
                               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef):
            top |= {m.name for m in n.body
                    if isinstance(m, (ast.FunctionDef, ast.Assign))
                    and hasattr(m, 'name')}
    # Only analyze OUTERMOST functions here; analyze() recurses into nested
    # ones carrying the enclosing scope, so a closure legitimately reading an
    # enclosing local is not a finding.
    nested = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.Lambda)):
            for c in ast.walk(n):
                if c is not n and isinstance(c, (ast.FunctionDef, ast.Lambda)):
                    nested.add(id(c))
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and id(n) not in nested:
            analyze(n, top, path, bad)
    return bad

def check_methods(path):
    """Flag self.<name>(...) calls with no matching def in the class.

    A missing METHOD is an AttributeError, not a scope error, so the analysis
    above cannot see it - and it is the natural failure mode when refactoring
    splits or merges methods: the call site outlives the definition.
    """
    tree = ast.parse(open(path).read())
    bad = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        defined = set()
        for n in ast.walk(cls):
            if isinstance(n, ast.FunctionDef):
                defined.add(n.name)
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        defined.add(t.id)
                    elif (isinstance(t, ast.Attribute)
                          and isinstance(t.value, ast.Name)
                          and t.value.id == 'self'):
                        defined.add(t.attr)
        # Inherited members are unknowable statically; only flag when the class
        # has no bases, where every self.<name> must be defined here.
        if cls.bases:
            continue
        for n in ast.walk(cls):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == 'self'
                    and n.func.attr not in defined):
                bad.append((path, cls.name, n.lineno, n.func.attr))
    return bad

issues = []
for p in sys.argv[1:]:
    issues += check(p)
    for path, cls, ln, attr in check_methods(p):
        print("%s:%d  %s.self.%s() is called but never defined"
              % (path, ln, cls, attr))
        issues.append((path, cls, ln, attr))
seen = set()
for path, fn, ln, name in issues:
    if (fn, name) in seen: continue
    seen.add((fn, name))
    print("%s:%d  %s() uses undefined local '%s'" % (path, ln, fn, name))
# (exit moved to the end so later checks still run)

arity_bad = []

# --- tuple-arity heuristic -------------------------------------------------
# A separate failure mode from scope: changing what a function appends to a
# list breaks every `for (a, b, c) in that_list` elsewhere with a runtime
# ValueError.  py_compile and the scope pass both miss it.  Flag destructuring
# loops whose arity disagrees with another destructuring of the SAME name in
# the same file - cheap, and it catches the real case.
def check_arity(path):
    tree = ast.parse(open(path).read())
    seen = {}
    bad = []
    for n in ast.walk(tree):
        if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple):
            src = None
            it = n.iter
            if isinstance(it, ast.Name):
                src = it.id
            elif isinstance(it, ast.comprehension):
                continue
            if src:
                k = (path, src)
                arity = len(n.target.elts)
                if k in seen and seen[k][0] != arity:
                    bad.append((path, n.lineno, src, seen[k][0], arity))
                else:
                    seen.setdefault(k, (arity, n.lineno))
    return bad

for p in sys.argv[1:]:
    for path, ln, name, a, b in check_arity(p):
        print("%s:%d  '%s' destructured as %d here but %d elsewhere"
              % (path, ln, name, b, a))
        arity_bad.append((path, ln, name))

sys.exit(1 if (issues or arity_bad) else 0)
