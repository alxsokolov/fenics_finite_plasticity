import numpy as np
import ufl
import math
from dolfinx import fem, mesh, log
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py.PETSc import ScalarType, Options


N=30
domain = mesh.create_box(MPI.COMM_WORLD,[[0.0,0.0,-1.0], [1.0, 1.0, 0.0]], [N, N, N//2], mesh.CellType.hexahedron)
x = ufl.SpatialCoordinate(domain)

def top(x):
    return np.isclose(x[2], 0)

def symmetry_x(x):
    return np.isclose(x[0], 0)

def symmetry_y(x):
    return np.isclose(x[1], 0)

def bottom(x):
    return np.isclose(x[2], -1.0)

def wall_x(x):
     return np.isclose(x[0], 1.0)

def wall_y(x):
     return np.isclose(x[1], 1.0)

tdim = domain.topology.dim
fdim = tdim - 1

bottom_facets = mesh.locate_entities_boundary(domain,fdim,bottom)
top_facets = mesh.locate_entities_boundary(domain,fdim,top)
symx_facets = mesh.locate_entities_boundary(domain, fdim, symmetry_x)
symy_facets = mesh.locate_entities_boundary(domain, fdim, symmetry_y)
wallx_facets = mesh.locate_entities_boundary(domain, fdim, wall_x)
wally_facets = mesh.locate_entities_boundary(domain, fdim, wall_y)

marked_facets = np.hstack([bottom_facets, top_facets, symx_facets, symy_facets, wallx_facets, wally_facets])
marked_values = np.hstack([np.full_like(bottom_facets, 1), np.full_like(top_facets, 2), np.full_like(symx_facets, 3), \
                           np.full_like(symy_facets, 4), np.full_like(wallx_facets, 5), np.full_like(wally_facets, 6)])
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

metadata = {"quadrature_degree":4}
ds = ufl.Measure('ds',domain=domain, subdomain_data=facet_tag, metadata=metadata)

import basix.ufl as bufl
gdim = domain.geometry.dim
V = fem.functionspace(domain, bufl.element("Lagrange", domain.basix_cell(), 1, shape=(gdim,)))
V2 = fem.functionspace(domain, ("Lagrange", 1))
V0 = fem.functionspace(domain, ("DG", 0))

u = fem.Function(V, name="Displacement")
du = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)
gap = fem.Function(V2, name="Gap")
p = fem.Function(V0, name="Contact pressure")

u_zero = np.array((0,)*domain.geometry.dim, dtype=ScalarType)
symx_dof = fem.locate_dofs_topological(V.sub(0), fdim, symx_facets)
symy_dof = fem.locate_dofs_topological(V.sub(1), fdim, symy_facets)

bc1 = fem.dirichletbc(u_zero, fem.locate_dofs_topological(V, fdim, facet_tag.find(1)), V)
bc2 = fem.dirichletbc(ScalarType(0), symx_dof, V.sub(0))
bc3 = fem.dirichletbc(ScalarType(0), symy_dof, V.sub(1))
bc = [bc1, bc2, bc3]

R = 0.5
d = 0.02

circle = -d+(pow(x[0], 2)+pow(x[1], 2))/2/R

E = fem.Constant(domain, ScalarType(10.))
nu = fem.Constant(domain, ScalarType(0.3))
mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)

#def eps(v):
#    return ufl.sym(ufl.grad(v))

def deformation_gradient(v):
    return ufl.Identity(3) + ufl.grad(v)

def eps(v):
    F = deformation_gradient(v)
    return 0.5*(ufl.Identity(3) - ufl.inv(ufl.dot(F, F.T)))

def sigma(v):
    return lmbda*ufl.tr(eps(v))*ufl.Identity(3) + 2.0*mu*eps(v)

def ppos(x):
    return ufl.max_value(x, 0)
pen = fem.Constant(domain, ScalarType(1e5))

form = ufl.inner(sigma(u), ufl.sym(ufl.grad(u_)))*ufl.dx + pen * \
    ufl.dot(u_[2], ppos(u[2]-circle))*ds(2)
J = ufl.derivative(form, u, du)

problem = NonlinearProblem(form, u, bc, J=J)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
ksp = solver.krylov_solver
opts = Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

log.set_log_level(log.LogLevel.INFO)
n, converged = solver.solve(u)
assert(converged)
print(f"Number of interations: {n:d}")

sig = sigma(u)[2, 2]
stress_expr = fem.Expression(sig, V0.element.interpolation_points())
stresses = fem.Function(V0)
stresses.interpolate(stress_expr)
maxstress = max(np.abs(stresses.x.array))

gap = circle-u[2]
gap_expr = fem.Expression(gap, V2.element.interpolation_points())
gapval = fem.Function(V2)
gapval.interpolate(gap_expr)
maxgap = max(np.abs(gapval.x.array))

a = math.sqrt(R*d)
F = 4/3.*float(E)/(1-float(nu)**2)*a*d
p0 = 3*F/(2*math.pi*a**2)

with XDMFFile(MPI.COMM_WORLD, "cube_dolfinx.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tag, domain.geometry)
    xdmf.write_function(u, 0.)

print(f'Contactarea:           {a:.3f} mm²')
print(f'Force:                 {F:.3f} N')
print(f'max Pressure:          {p0:.3f} MPa')
print(f'Computed max Pressure: {maxstress:.3} MPa')