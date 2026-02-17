#%%
import sympy as sp

# %%
q = sp.Symbol('q')
s = sp.Symbol('s')
c = sp.Symbol('c')
h = sp.Symbol('h')

expr_gd = (q**2 * (1-s) + q * (1-q) * (1+c) * (1 - h*s)) / (q**2 * (1-s) + 2 * q * (1-q) * (1 - h*s) + (1-q)**2)

# %%
q1, q2, q3 = sp.solvers.solve(expr_gd - q, q)

# Boundaries where internal root collides with 0 or 1:
s_q3_0 = sp.solve(sp.simplify(q3), s)          # list of s(c,h)
s_q3_1 = sp.solve(sp.simplify(q3 - 1), s)      # list of s(c,h)
c_q3_0 = sp.solve(sp.simplify(q3), c)
c_q3_1 = sp.solve(sp.simplify(q3 - 1), c)  
#%%
# find dq and evaluate at q3=eq, if > 1 then unstable 
df_dq = sp.diff(expr_gd, q) 
df_at_q3 = sp.simplify(df_dq.subs(q, q3))
# Boundaries where internal root is neutrally stable:  f'(q*) = ±1
s_at_df_eq_pos1 = sp.solve(sp.Eq(df_at_q3, 1), s)    # list of s(c,h)
s_at_df_eq_neg1 = sp.solve(sp.Eq(df_at_q3, -1), s)   # list of s(c,h)
c_at_df_eq_pos1 = sp.solve(sp.Eq(df_at_q3, 1), c)    # list of s(c,h)
c_at_df_eq_neg1 = sp.solve(sp.Eq(df_at_q3, -1), c)   # list of s(c,h)

#%%
# Lambdify as functions of (c,h); we’ll filter numerically later
def _lamb_list(expr_list):
    funs = []
    for expr in expr_list:
        funs.append(sp.lambdify((c,h), sp.simplify(expr), 'numpy'))
    return funs

def _lamb_list_c(expr_list):
    funs = []
    for expr in expr_list:
        funs.append(sp.lambdify((s,h), sp.simplify(expr), 'numpy'))
    return funs

s_q3_0_funcs = _lamb_list(s_q3_0)
s_q3_1_funcs = _lamb_list(s_q3_1)
s_df_pos1_funcs = _lamb_list(s_at_df_eq_pos1)
s_df_neg1_funcs = _lamb_list(s_at_df_eq_neg1)

c_q3_0_funcs = _lamb_list_c(c_q3_0)
c_q3_1_funcs = _lamb_list_c(c_q3_1)
c_df_pos1_funcs = _lamb_list_c(c_at_df_eq_pos1)
c_df_neg1_funcs = _lamb_list_c(c_at_df_eq_neg1)


#%%
# Also lambdas to evaluate q3 and f'(q3) at numeric (s,c,h)
q3_func      = sp.lambdify((s,c,h), sp.simplify(q3), 'numpy')
df_at_q3_func= sp.lambdify((s,c,h), df_at_q3, 'numpy')

s_q3_0_lambda = sp.lambdify((c, h), s_q3_0, 'numpy')
s_q3_1_lambda = sp.lambdify((c, h), s_q3_1, 'numpy')
# %%


def find_state(s, c, h):
    """
    Analytically determine the state given (s, c, h) values.

    Parameters:
        s (float): Selection coefficient.
        c (float): Cost coefficient.
        h (float): Dominance coefficient.

    Returns:
        str: The state of the system ('stable', 'unstable', or 'boundary').
    """
    # Evaluate q3 and f'(q3) at the given (s, c, h)
    try:
        q3_val = q3_func(s, c, h)
        df_at_q3_val = df_at_q3_func(s, c, h)
    except Exception as e:
        return f"Error in evaluation: {e}"

    # Check if q3 is within the valid range [0, 1]
    if not (0 <= q3_val <= 1):
        return "boundary"

    # Check stability based on f'(q3)
    if abs(df_at_q3_val) < 1:
        return "stable"
    elif abs(df_at_q3_val) > 1:
        return "unstable"
    else:
        return "boundary"