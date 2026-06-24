import numpy as np
import os
import ufl
import math
from dolfinx import fem, mesh, log
from dolfinx.fem.petsc import NewtonSolverNonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.io import XDMFFile
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

# Avoid JAX preallocating all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# --- Import JAX AFTER setting env vars ---
import jax

import jaxmat.materials as jm
from dolfinx_materials.jaxmat import JAXMaterial
from dolfinx_materials.quadrature_map import QuadratureMap
from dolfinx_materials.solvers import NonlinearMaterialProblem
from dolfinx_materials.utils import nonsymmetric_tensor_to_vector
from dolfinx_materials.quadrature_function import create_quadrature_function

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

def F(u):
    return nonsymmetric_tensor_to_vector(ufl.Identity(3) + ufl.grad(u))


def dF(u, v):
    return ufl.derivative(F(u), u, v)

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
order = 2
deg_quad = 2 * (order - 1)
V = fem.functionspace(domain, bufl.element("Lagrange", domain.basix_cell(), order, shape=(gdim,)))
V2 = fem.functionspace(domain, ("Lagrange", order))
V0 = fem.functionspace(domain, ("DG", order - 1))

u = fem.Function(V, name="Displacement")
du = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
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
d = 0.002

circle = -d+(pow(x[0], 2)+pow(x[1], 2))/2/R

E = 70e3
nu = 0.3
sig0 = 500.0

b = 1000
sigu = 750.0
   
#E = fem.Constant(domain, ScalarType(10.))
#nu = fem.Constant(domain, ScalarType(0.3))
#mu = E/2/(1+nu)
#lmbda = E*nu/(1+nu)/(1-2*nu)

def ppos(x):
    return ufl.max_value(x, 0)
pen = fem.Constant(domain, ScalarType(1e5))

elasticity = jm.LinearElasticIsotropic(E=E, nu=nu)

hardening = jm.VoceHardening(sig0=sig0, sigu=sigu, b=b)

behavior = jm.FeFpJ2Plasticity(elasticity=elasticity, yield_stress=hardening)

material = JAXMaterial(behavior)

qmap = QuadratureMap(domain, deg_quad, material)
qmap.register_gradient("F", F(u))

P = qmap.fluxes["PK1"]
Res = ufl.dot(P, dF(u, v)) * qmap.dx + pen * ufl.dot(v[2], ppos(u[2]-circle))*ds(2)
Jac = qmap.derivative(Res, u, du)

qmap.update()


#def eps(v):
#    return ufl.sym(ufl.grad(v))

# def deformation_gradient(v):
#     return ufl.Identity(3) + ufl.grad(v)

# def eps(v):
#     F = deformation_gradient(v)
#     return 0.5*(ufl.Identity(3) - ufl.inv(ufl.dot(F, F.T)))

# def sigma(v):
#     return lmbda*ufl.tr(eps(v))*ufl.Identity(3) + 2.0*mu*eps(v)




petsc_options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "snes_atol": 1e-8,
    "snes_rtol": 1e-8,
    "snes_max_it": 20,
    "snes_monitor": "",
    "snes_converged_reason": "",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
problem = NonlinearMaterialProblem(
    qmap,
    Res,
    u,
    bcs=bc,
    J=Jac,
    petsc_options_prefix="elastoplasticity",
    petsc_options=petsc_options,
)

problem.solve()
converged = problem.solver.getConvergedReason()
num_iter = problem.solver.getIterationNumber()

# sig = sigma(u)[2, 2]
# stress_expr = fem.Expression(sig, V0.element.interpolation_points)
# stresses = fem.Function(V0)
# stresses.interpolate(stress_expr)
# maxstress = max(np.abs(stresses.x.array))

# gap = circle-u[2]
# gap_expr = fem.Expression(gap, V2.element.interpolation_points)
# gapval = fem.Function(V2)
# gapval.interpolate(gap_expr)
# maxgap = max(np.abs(gapval.x.array))

# a = math.sqrt(R*d)
# F = 4/3.*float(E)/(1-float(nu)**2)*a*d
# p0 = 3*F/(2*math.pi*a**2)

V1_out = fem.functionspace(domain, bufl.element("Lagrange", domain.basix_cell(), 1, shape=(gdim,)))
u_out = fem.Function(V1_out, name="Displacement")
u_out.interpolate(u)

# Von Mises stress: evaluate at quadrature points then average to DG0
P_fn = qmap.fluxes["PK1"]
Fdef = ufl.Identity(3) + ufl.grad(u)
J_det = ufl.det(Fdef)
P_tens = ufl.as_tensor([
    [P_fn[0], P_fn[3], P_fn[5]],
    [P_fn[4], P_fn[1], P_fn[7]],
    [P_fn[6], P_fn[8], P_fn[2]],
])
sigma_c = (1 / J_det) * P_tens * Fdef.T
s_dev = sigma_c - (1/3) * ufl.tr(sigma_c) * ufl.Identity(3)
vm_expr_ufl = ufl.sqrt(3/2 * ufl.inner(s_dev, s_dev))

n_qp = len(qmap.quadrature_points)
vm_quad = create_quadrature_function("VonMises_quad", 1, domain, deg_quad)
qmap.eval_quadrature(vm_expr_ufl, vm_quad)

W0 = fem.functionspace(domain, ("DG", 0))
vm_out = fem.Function(W0, name="VonMises")
vm_out.x.array[:] = vm_quad.x.array.reshape(-1, n_qp).mean(axis=1)

p_fn = qmap.internal_state_variables["p"]
p_out = fem.Function(W0, name="PlasticStrain")
p_out.x.array[:] = p_fn.x.array.reshape(-1, n_qp).mean(axis=1)

with XDMFFile(MPI.COMM_WORLD, "cube_dolfinx.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_meshtags(facet_tag, domain.geometry)
    xdmf.write_function(u_out, 0.)
    xdmf.write_function(vm_out, 0.)
    xdmf.write_function(p_out, 0.)

# print(f'Contactarea:           {a:.3f} mm²')
# print(f'Force:                 {F:.3f} N')
# print(f'max Pressure:          {p0:.3f} MPa')
# print(f'Computed max Pressure: {maxstress:.3} MPa')