
#### Version 3: Taylor expansion around q = 0 ######################
#%%
import sympy
from sympy import pprint

q = sympy.Symbol('q')
s = sympy.Symbol('s')
c = sympy.Symbol('c')
h = sympy.Symbol('h')

s2 = sympy.Symbol('s2')
h2 = sympy.Symbol('h2')

# %%
expr_gd = (q**2 * (1-s) + q * (1-q) * (1+c) * (1 - h*s)) / (q**2 * (1-s) + 2 * q * (1-q) * (1 - h*s) + (1-q)**2)
expr_ngd = (q**2 * (1-s2) + q * (1-q) * (1 - h2*s2)) / (q**2 * (1-s2) + 2 * q * (1-q) * (1 - h2*s2) + (1-q)**2)

# %%
# simplify the expression and solve it for s2 and h2
# eq is the s2 = ... expression in terms of s, c, h. and q
sympy.simplify(expr_gd - expr_ngd)
eq = sympy.solvers.solve(expr_gd - expr_ngd, s2)[0]
# pprint(eq)
eq_sub = eq.subs(s, 0.1).subs(c, 0.9).subs(h, 0.8).subs(h2, 0.18)
# print("\nEquation after substituting s, c, h, h2:")
# pprint(eq_sub)
# eq_sub = eq_sub.subs(q, 0.9)
# print("\nEquation after substituting q:")
# pprint(eq_sub)

#%%
# Extract the denominator of the current equation
denominator = sympy.denom(eq_sub)
# print("\nDenominator of the equation:")
# pprint(denominator)

# Solve for the value of q when the denominator equals 0
q_values = sympy.solvers.solve(denominator, q)
# print("\nValues of q when the denominator equals 0:")
# pprint(q_values)

#%%
# taylor expansion of equation around q = 0 ##############################
eq_taylor = sympy.series(eq, q, 0, 3)
# print("Taylor expansion of the equation around q = 0:")
# pprint(eq_taylor)

# a0 is the constant term, a1 is the coefficient of q
# sngd - a0 - a1 * q = 0
a0 = eq_taylor.coeff(q, 0)
a1 = eq_taylor.coeff(q, 1)
a2 = eq_taylor.coeff(q, 2)

# substitute sngd and hngd into a2 to check the value of the second order term

# solve for s2 and h2 by solving the system of equations: 
# 1) sngd - a0 = 0
# 2) a1 = 0
s2_solved, h2_solved = sympy.solvers.solve([s2-a0, a1], [s2,h2])[0]


print("Symbolic expression for s2:")
pprint(s2_solved)

print("\nSymbolic expression for h2:")
pprint(h2_solved)

#%%
### Unstable #############################################################
print("\nUnstable case:")
eq = (2*0.5*(1-c)*(1-h*s)+2*c*(1-h*s)-1)/(4*0.5*(1-c)*(1-h*s)+2*c*(1-h*s)+s-2)
h_eq = sympy.simplify(eq/(2*eq-1))
a0_h = a0.subs(h2, h_eq)
s2_unstable_solved = sympy.solve(s2-a0_h, s2)[0]
print("\nSymbolic expression for s2 with calculated h2:")
pprint(s2_unstable_solved)
pprint(h_eq)

#### Solving for s and h at specific values of c ##########################
#%%
# s2_c0, h2_c0 = s2_solved.subs(c, 0), h2_solved.subs(c, 0)
# print("\nSolved values for s2 and h2 at c = 0:")
# pprint(s2_c0)
# pprint(h2_c0)
# s2_c05, h2_c05 = s2_solved.subs(c, 0.5), h2_solved.subs(c, 0.5)
# print("\nSolved values for s2 and h2 at c = 0.5:")
# pprint(s2_c05)
# pprint(h2_c05)
# print("\nSolved values for s2 and h2 at c = 1.0:")
# s2_c1, h2_c1 = s2_solved.subs(c, 1.0), h2_solved.subs(c, 1.0)
# pprint(s2_c1)
# pprint(h2_c1)

#%%
# check the value of a2 ##################################
# a2 = a2.subs(s2, s2_solved).subs(h2, h2_solved)
# pprint("Second order term (a2) in terms of s, c, h:")
# pprint(a2)
# sv, cv, hv = 0.1, 0.09, 0.8
# print(f"\nSecond order term (a2) value with s={sv}, c={cv}, h={hv}:")
# a2 = a2.subs(s, sv).subs(c, cv).subs(h, hv)
# pprint(a2)


#%%
s2_func = sympy.lambdify((s, c, h), s2_solved, modules='numpy')
h2_func = sympy.lambdify((s, c, h), h2_solved, modules='numpy')
h2_unstable_func = sympy.lambdify((s, c, h), h_eq, modules='numpy')
s2_unstable_func = sympy.lambdify((s, c, h), s2_unstable_solved, modules='numpy')

# print(s2_solved.subs(s, 0.2).subs(c,0.95).subs(h,0.3))
# print(h2_solved.subs(s, 0.2).subs(c,0.95).subs(h,0.3))

# %%
def solve_sngd(s, c, h):
    """
    Solve the equation for s2 given s, c, h.
    
    Parameters:
    s (float): The value of s.
    c (float): The value of c.
    h (float): The value of h.
    h2 (float): The value of h2.
    
    Returns:d
    float: The solution for s2.
    """
    return s2_func(s, c, h), h2_func(s, c, h)

def solve_sngd_unstable(s, c, h):
    return s2_unstable_func(s, c, h), h2_unstable_func(s, c, h)

# %%
### Version 2: Substitute q with specific values and solve for s2 and h2 ##########

# q1 = sp.Rational(200, 1000)
# q2 = sp.Rational(800, 1000)

# eq1 = expr_gd.subs(q, q1) - expr_ngd.subs(q, q1)
# eq2 = expr_gd.subs(q, q2) - expr_ngd.subs(q, q2)

# sol = sp.solve([eq1, eq2], [s2, h2], dict=True)

# s2_sol = sp.simplify(sol[0][s2])
# h2_sol = sp.simplify(sol[0][h2])

# # print("Symbolic expression for s2:")
# # pprint(s2_sol)

# # print("\nSymbolic expression for h2:")
# # pprint(h2_sol)


# s2_func = sp.lambdify((s, c, h), s2_sol, modules='numpy')
# h2_func = sp.lambdify((s, c, h), h2_sol, modules='numpy')
# # %%

# # def solve_sngd(s, c, h):
# #     """
# #     Solve the equation for s2 given s, c, h, and h2.
    
# #     Parameters:
# #     s (float): The value of s.
# #     c (float): The value of c.
# #     h (float): The value of h.
# #     h2 (float): The value of h2.
    
# #     Returns:
# #     float: The solution for s2.
# #     """
# #     return s2_func(s, c, h), h2_func(s, c, h)

# # print(solve_sngd(0.5, 0.5, 0.3))